import copy
import csv
import json
import os
import torch
import random
import fvdb.nn as fvnn
from typing import *
import torch.nn as nn
from contextlib import contextmanager
from diffusers.training_utils import (
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
)
from transformers import CLIPTextModel, AutoTokenizer
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import (
    retrieve_timesteps,
)

from tqdm import tqdm
from seen2scene.tools.ema import LitEma
import seen2scene.tools.common_utils as cmt
import seen2scene.tools.vdb_utils as vdb_utils
from seen2scene.tools.log_utils import get_logger
from seen2scene.models.fields import split_fvdb, merge_fvdb
from seen2scene.models.sparse.attention.modules import RotaryPositionEmbedder
from seen2scene.models.trainer.common import get_instance_mask

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
from seen2scene.tools.loss_utils import masked_weighted_loss
from seen2scene.models import sparse as sp

logger = get_logger(file_name=__file__, debug="flow_matching")


def get_sigmas(
    scheduler,
    timesteps: torch.Tensor,
    n_dim: int = 2,
    dtype: torch.dtype = torch.float32,
    device: str = "cuda",
) -> torch.Tensor:
    """Retrieve scheduler sigmas aligned to provided timesteps.

    Args:
        scheduler: The diffusion scheduler providing `sigmas` and `timesteps`.
        timesteps: 1D tensor of selected timesteps, device-agnostic.
        n_dim: Number of trailing singleton dims to append for broadcasting.
        dtype: Desired dtype for the resulting sigma tensor.

    Returns:
        A tensor of sigmas broadcastable to the target shape.
    """
    sigmas = scheduler.sigmas.to(device=device, dtype=dtype)
    schedule_timesteps = scheduler.timesteps.to(device)
    step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
    sigma = sigmas[step_indices].flatten()
    while len(sigma.shape) < n_dim:
        sigma = sigma.unsqueeze(-1)
    return sigma


class Conditioner(nn.Module):
    """Semantic conditioning module using CLIP embeddings for object-based control.

    Encodes object names and optionally spatial distance information into conditioning
    features for the diffusion model. Uses pre-computed CLIP embeddings for known
    semantic classes and computes embeddings on-the-fly for unknown object names.

    When use_distance_rope=True, encodes normalized distances from voxels to object
    bounding box centers using sinusoidal position embeddings, allowing the model
    to be aware of spatial relationships between voxels and objects. Uses SPARSE
    computation to only process non-zero ins_mask entries, achieving ~24x memory
    reduction compared to dense implementation.

    Attributes:
        model_channels: Model hidden dimension for conditioning projection.
        use_distance_rope: Whether to use positional encoding for distance conditioning.
        clip: Dictionary containing CLIP tokenizer and text encoder model.
        clip_embeddings: Pre-computed CLIP embeddings for semantic class names.
        name_to_idx: Mapping from class names to embedding indices.
        proj: Projection layer to map CLIP embeddings to model_channels dimension.
        num_pos_features: Number of positional encoding features (equals model_channels).
    """

    def __init__(
        self,
        model_channels: int,
        use_distance_rope: bool = False,
        use_label_augmentation: bool = False,
    ):
        """Initialize the conditioner with CLIP model and semantic class embeddings.

        Args:
            model_channels: Hidden dimension of the diffusion model.
            use_distance_rope: If True, use RoPE to encode normalized distances from
                voxels to object bounding box centers.
            use_label_augmentation: If True, randomly replace object labels with
                pre-computed synonyms from the mapping CSVs during training.
        """
        super().__init__()
        self.model_channels = model_channels
        self.use_distance_rope = use_distance_rope
        self.use_label_augmentation = use_label_augmentation
        if use_label_augmentation:
            self._synonym_map: Dict[str, List[str]] = self._load_synonym_map()
        # Load semantic class names for precomputed embeddings
        from seen2scene import ASSETS_DIR

        csv_path = str(ASSETS_DIR / "semantic_classes.csv")
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            class_names = [row["name"] for row in reader]
        class_names += [" "]

        # Load CLIP model and tokenizer
        text_model_name = "openai/clip-vit-base-patch32"
        self.clip = {
            "tokenizer": AutoTokenizer.from_pretrained(text_model_name),
            "model": CLIPTextModel.from_pretrained(
                text_model_name, use_safetensors=True
            ),
        }
        self.clip["model"].requires_grad_(False)
        self.clip["model"].eval()
        clip_dim = self.clip["model"].config.projection_dim

        # Precompute CLIP embeddings for known semantic class names
        with torch.no_grad():
            encoding = self.clip["tokenizer"](
                class_names,
                max_length=77,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            embeddings = self.clip["model"](
                input_ids=encoding["input_ids"]
            ).pooler_output

        self.register_buffer(
            "clip_embeddings", embeddings, persistent=False
        )  # [N_classes, 512]
        self.name_to_idx: Dict[str, int] = {
            name: idx for idx, name in enumerate(class_names)
        }

        # Cache for dynamically computed embeddings (for raw/unknown names)
        self._embedding_cache: Dict[str, torch.Tensor] = {}

        self.proj = nn.Sequential(nn.Linear(clip_dim, model_channels), nn.SELU(True))

        # RoPE for encoding normalized distances from voxels to object bbox centers
        if self.use_distance_rope:
            self.distance_rope = RotaryPositionEmbedder(
                hidden_size=model_channels, in_channels=3  # 3D coordinates (x, y, z)
            )

    def _apply_rope_directly(
        self, features: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply RoPE to features using positions.

        RotaryPositionEmbedder expects x to have at least 3 dimensions [N, H, D]
        where H is the number of heads. Since we have 2D features [K, C], we add
        a singleton head dimension [K, 1, C] for compatibility.

        Args:
            features: [K, C] sparse features
            positions: [K, 3] 3D positions (normalized)

        Returns:
            [K, C] RoPE-encoded features
        """
        # Add head dimension: [K, C] -> [K, 1, C]
        features_3d = features.unsqueeze(1)

        # Call RoPE: [K, 1, C], [K, 3] -> [K, 1, C]
        rope_features_3d = self.distance_rope(features_3d, positions)

        # Remove head dimension: [K, 1, C] -> [K, C]
        rope_features = rope_features_3d.squeeze(1)

        return rope_features

    @staticmethod
    def _load_synonym_map() -> Dict[str, List[str]]:
        """Load pre-computed synonyms from assets/label_synonyms.json."""
        from seen2scene import ASSETS_DIR

        json_path = str(ASSETS_DIR / "label_synonyms.json")
        if not os.path.exists(json_path):
            logger.warning(
                f"Synonym map not found at {json_path}. Run: python tools/augment_labels.py"
            )
            return {}
        with open(json_path, encoding="utf-8") as f:
            synonym_map = json.load(f)
        synonym_map = {k: v for k, v in synonym_map.items() if v}
        logger.info(
            f"Loaded synonym map with {len(synonym_map)} entries from {json_path}"
        )
        return synonym_map

    def augment_names(self, names: List[List[str]]) -> List[List[str]]:
        """Randomly replace each label with a pre-computed synonym (training only).

        Synonyms are loaded from aug_* columns in the mapping CSVs (written by
        tools/augment_labels.py).  Labels absent from the map are kept unchanged.
        """
        if not self.use_label_augmentation:
            return names
        return [
            [
                (
                    random.choice(self._synonym_map.get(n, [n]) + [n])
                    if random.random() < 0.5
                    else n
                )
                for n in batch
            ]
            for batch in names
        ]

    def to(self, *args, **kwargs):
        """Override to() to ensure CLIP model is also moved."""
        self = super().to(*args, **kwargs)
        self.clip["model"] = self.clip["model"].to(*args, **kwargs)
        return self

    def cuda(self, device=None):
        """Override cuda() to ensure CLIP model is also moved."""
        self = super().cuda(device)
        self.clip["model"] = self.clip["model"].cuda(device)
        return self

    def cpu(self):
        """Override cpu() to ensure CLIP model is also moved."""
        self = super().cpu()
        self.clip["model"] = self.clip["model"].cpu()
        return self

    @torch.no_grad()
    def get_clip_embedding(self, names: List[str]) -> torch.Tensor:
        """Get CLIP text embeddings for object names using pre-computed cache or on-the-fly computation.

        Efficiently retrieves CLIP embeddings by:
        1. Using pre-computed embeddings for known semantic class names
        2. Checking runtime cache for previously computed unknown names
        3. Computing new embeddings only for uncached unknown names

        Args:
            names: List of object name strings.

        Returns:
            Tensor of CLIP embeddings with shape [len(names), 512].

        Notes:
            - Pre-computed embeddings are stored for ~150 semantic classes
            - Unknown names are computed once and cached for future use
            - All computations are done without gradients (@torch.no_grad())
        """
        device = self.clip_embeddings.device
        embeddings: List[torch.Tensor] = []
        pending: List[str] = []
        pending_indices: List[int] = []

        for idx, name in enumerate(names):
            if name in self.name_to_idx:
                embeddings.append(self.clip_embeddings[self.name_to_idx[name]])
                continue

            # Check cache for previously computed embeddings
            if name in self._embedding_cache:
                cached = self._embedding_cache[name]
                if cached.device != device:
                    cached = cached.to(device)
                    self._embedding_cache[name] = cached
                embeddings.append(cached)
                continue

            embeddings.append(None)
            pending.append(name)
            pending_indices.append(idx)

        if pending:
            encoding = self.clip["tokenizer"](
                pending,
                max_length=77,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            batch_embeddings = self.clip["model"](
                input_ids=encoding["input_ids"].to(device)
            ).pooler_output

            for i, name in enumerate(pending):
                emb = batch_embeddings[i]
                self._embedding_cache[name] = emb
                embeddings[pending_indices[i]] = emb

        # Fix: Validate embeddings before stacking
        valid_embeddings = [e for e in embeddings if e is not None]
        if len(valid_embeddings) == 0:
            raise ValueError(
                "No valid embeddings found - all object names may be invalid"
            )
        if len(valid_embeddings) != len(embeddings):
            raise ValueError(
                f"Some embeddings are None: {len(valid_embeddings)}/{len(embeddings)} valid"
            )
        return torch.stack(valid_embeddings)

    def forward(
        self,
        voxel_coords: torch.Tensor,
        object_names: List[List[str]],
        ins_mask: Optional[torch.Tensor],
        object_bboxes: Optional[List[List[torch.Tensor]]],
    ) -> sp.SparseTensor:
        """Forward pass with optional distance-based RoPE conditioning.

        Uses sparse computation when use_distance_rope=True: only processes non-zero
        ins_mask entries instead of all (voxel, object) pairs. This reduces memory
        from O(N*L*C) to O(K*C) where K is the number of non-zero ins_mask entries,
        typically K ≈ N*avg_objects_per_voxel << N*L.

        Args:
            voxel_coords: coordinates
            object_names: List of object names per batch.
            ins_mask: Instance mask [N, L] indicating which objects each voxel belongs to.
                Typically sparse: each voxel belongs to 1-3 objects out of L total.
            object_bboxes: list of object bounding boxes per batch. Each bbox
                is a tensor of shape [2, 3] with [[xmin, ymin, zmin], [xmax, ymax, zmax]].
                Required when use_distance_rope=True.

        Returns:
            Sparse tensor with conditioning features.

        Notes:
            Memory usage comparison (N=20k voxels, L=100 objects, C=256 channels):
            - Dense (old): ~4 GB (stores [N, L, C] tensors)
            - Sparse (new): ~80 MB (stores [K, C] tensors, K ≈ 40k)
        """
        device = voxel_coords.device
        flat_names = [x for names in object_names for x in names]
        embeds = self.get_clip_embedding(flat_names)  # [L, 512]
        embeds = self.proj(embeds)  # [L, C]

        if self.use_distance_rope and object_bboxes is not None:
            # Sparse computation: only process non-zero ins_mask entries
            # This reduces memory from O(N*L*C) to O(K*C) where K << N*L
            # Use world-space voxel centers to match object_bboxes coordinates.
            N = voxel_coords.shape[0]

            # Get sparse indices where ins_mask is non-zero
            # nz_voxel_idx[i] = voxel index, nz_obj_idx[i] = object index
            nz_voxel_idx, nz_obj_idx = ins_mask.nonzero(as_tuple=True)
            K = len(nz_voxel_idx)  # Number of sparse entries

            if K == 0:
                # No voxels belong to any objects - return zero features
                return torch.zeros(N, self.model_channels, device=device)

            # Extract embeddings only for objects that appear in sparse entries
            sparse_embeds = embeds[nz_obj_idx]  # [K, C]

            # Extract voxel coordinates only for voxels that belong to objects
            sparse_voxel_coords = voxel_coords[nz_voxel_idx]  # [K, 3]

            # Flatten object bboxes from List[List[Tensor]] to List[Tensor]
            flat_bboxes = [bbox for bboxes in object_bboxes for bbox in bboxes]
            bbox_tensor = torch.stack(flat_bboxes).to(device)  # [L, 2, 3]

            # Compute bbox centers and sizes for all objects
            bbox_min = bbox_tensor[:, 0, :]  # [L, 3]
            bbox_max = bbox_tensor[:, 1, :]  # [L, 3]
            bbox_centers = (bbox_min + bbox_max) / 2  # [L, 3]
            bbox_sizes = bbox_max - bbox_min  # [L, 3]

            # Extract centers and sizes only for objects in sparse entries
            sparse_centers = bbox_centers[nz_obj_idx]  # [K, 3]
            sparse_sizes = bbox_sizes[nz_obj_idx]  # [K, 3]

            # Compute normalized relative positions only for sparse entries
            # Each entry is: (voxel_i coords) - (object_j center) / (object_j size)
            rel_pos = sparse_voxel_coords - sparse_centers  # [K, 3]
            normalized_rel_pos = rel_pos * 2.0 / (sparse_sizes + 1e-6)  # [K, 3]

            # Apply RoPE to encode position information into embeddings
            # RoPE modulates the embeddings based on 3D positions
            rope_features = self._apply_rope_directly(
                sparse_embeds, normalized_rel_pos
            )  # [K, C]

            # Accumulate features back to [N, C] by scattering sparse entries
            # For each voxel, sum the position-encoded features of all objects it belongs to
            feats = torch.zeros(
                N, self.model_channels, device=device, dtype=rope_features.dtype
            )
            feats.index_add_(0, nz_voxel_idx, rope_features)

            # Normalize by the number of objects each voxel belongs to
            sum_i = ins_mask.float().sum(dim=-1, keepdim=True).clamp_min(1.0)  # [N, 1]
            feats = feats / sum_i  # [N, C]
        else:
            # Original implementation without distance conditioning
            ins_mask = ins_mask.float()  # [N, L]
            sum_i = ins_mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
            feats = (ins_mask @ embeds) / sum_i  # [N, C]

        return feats


class Model(nn.Module):
    def __init__(
        self,
        model_cfg: Dict[str, Any],
        scheduler: Dict[str, Any],
        num_inference_steps: int,
        cond_drop_prob: float = 0.1,
        precondition_outputs: bool = False,
        weighting_scheme: str = "logit_normal",
        logit_mean: float = 0.0,
        logit_std: float = 1.0,
        mode_scale: float = 1.29,
        use_ema: bool = False,
        band_weight: float = 0.5,
        zero_noise: bool = False,
        use_distance_rope: bool = False,
        use_label_augmentation: bool = False,
    ):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_inference_steps = num_inference_steps
        self.cond_drop_prob = cond_drop_prob
        self.precondition_outputs = precondition_outputs
        self.weighting_scheme = weighting_scheme
        self.logit_mean = logit_mean
        self.logit_std = logit_std
        self.mode_scale = mode_scale
        self.scheduler = cmt.instantiate_from_config(scheduler)
        self.use_ema = use_ema
        self.band_weight = band_weight
        assert 0 <= band_weight <= 1, "band_weight must be in (0, 1)"
        self.zero_noise = zero_noise
        self.use_distance_rope = use_distance_rope

        self.model = cmt.instantiate_from_config(model_cfg)
        if self.use_ema:
            self.model_ema = LitEma(self.model)
            logger.info(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        self.conditioner = Conditioner(
            model_channels=model_cfg["model_channels"],
            use_distance_rope=use_distance_rope,
            use_label_augmentation=use_label_augmentation,
        )

    @contextmanager
    def ema_scope(self):
        if self.use_ema:
            self.model_ema.store(self.model.parameters())
            self.model_ema.copy_to(self.model)
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.model.parameters())

    def losses(
        self,
        latent_tgt: fvnn.VDBTensor,
        object_names: List[List[str]],
        object_bboxes: Optional[List[List[torch.Tensor]]],
        band_mask: torch.Tensor,
        avg_over: str,
        ins_mask: Optional[torch.Tensor],
        latent_src: Optional[fvnn.VDBTensor] = None,
        controlnet: nn.Module = None,
        ema: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        device = latent_tgt.device
        batch_size = len(latent_tgt.grid)

        u = compute_density_for_timestep_sampling(
            weighting_scheme=self.weighting_scheme,
            batch_size=batch_size,
            logit_mean=self.logit_mean,
            logit_std=self.logit_std,
            mode_scale=self.mode_scale,
        )
        indices = (u * self.scheduler.config.num_train_timesteps).long()
        timesteps = self.scheduler.timesteps[indices].to(device=device)  # [B]
        sigmas = get_sigmas(
            self.scheduler, timesteps, dtype=torch.float32, device=device
        )

        if latent_src is not None:
            latent_known = latent_tgt.grid.fill_from_grid(
                latent_src.data, latent_src.grid, 0.0
            )
            latent_src = fvnn.VDBTensor(latent_tgt.grid, latent_known)
            latent_src = vdb_utils.vdb_to_spconv(latent_src)

        latent_tgt = vdb_utils.vdb_to_spconv(latent_tgt)
        sigmas = sigmas[latent_tgt.coords[:, 0].long()]  # [N, ...]
        noise = torch.randn_like(latent_tgt.feats)
        if self.zero_noise:
            noise = torch.zeros_like(noise)

        xt = (1.0 - sigmas) * latent_tgt.feats + sigmas * noise
        model_input = latent_tgt.replace(xt)

        # Label augmentation: randomly swap labels with LLM synonyms (training only)
        if self.training:
            object_names_ = self.conditioner.augment_names(object_names)
        else:
            object_names_ = object_names

        if random.random() < self.cond_drop_prob:
            object_bboxes = [
                torch.zeros((1, 2, 3), device=device) for _ in range(batch_size)
            ]
            object_names_ = [[" "] for _ in range(batch_size)]
            xyzs_list = [model_input[i].xyzs for i in range(model_input.shape[0])]
            ins_mask = get_instance_mask(xyzs_list, object_bboxes)

        context = self.conditioner(
            model_input.xyzs,
            object_names_,
            ins_mask=ins_mask,
            object_bboxes=object_bboxes,
        )

        controls = None
        if controlnet is not None:
            controls = controlnet(
                latent_src,
                timesteps,
                xt=model_input,
                embed=context,
                return_intermediates=True,
            )

        if not ema:
            model_output = self.model(
                model_input, timesteps, embed=context, controls=controls
            )
        else:
            with self.ema_scope():
                model_output = self.model(
                    model_input, timesteps, embed=context, controls=controls
                )

        # Follow: Section 5 of https://arxiv.org/abs/2206.00364.
        # Preconditioning of the model outputs.
        pred = model_output.feats
        if self.precondition_outputs:
            pred = pred * (-sigmas) + xt

        if self.precondition_outputs:
            target = latent_tgt.feats
        else:
            target = noise - latent_tgt.feats

        weighting = compute_loss_weighting_for_sd3(
            weighting_scheme=self.weighting_scheme, sigmas=sigmas
        )

        loss_stats = {}
        loss = masked_weighted_loss(
            pred, target, weighting=weighting, loss_type="l2", batch_reduction="none"
        )

        if avg_over == "region":
            assert band_mask is not None, "band_mask is required for region-level loss"

            loss_band = (loss * band_mask).sum() / (band_mask.sum() + 1e-6)
            loss_empty = (loss * (1 - band_mask)).sum() / ((1 - band_mask).sum() + 1e-6)
            loss = loss_band * self.band_weight + loss_empty * (1 - self.band_weight)

            loss_stats |= {
                "loss_band": loss_band.item(),
                "loss_empty": loss_empty.item(),
            }
        elif avg_over == "voxel":
            loss = loss.mean()
        else:
            raise NotImplementedError

        return loss, loss_stats

    @torch.no_grad()
    def sample(
        self,
        noise: fvnn.VDBTensor,
        object_names: List[List[str]],
        object_bboxes: Optional[List[List[torch.Tensor]]] = None,
        guidance_scale: float = 1.0,
        latent_src: Optional[fvnn.VDBTensor] = None,
        controlnet: Optional[nn.Module] = None,
    ) -> fvnn.VDBTensor:
        device = noise.device
        B = len(noise.grid)
        cfg = self.cond_drop_prob > 0 and guidance_scale != 1.0

        if latent_src is not None:
            latent_known = noise.grid.fill_from_grid(
                latent_src.data, latent_src.grid, 0.0
            )
            latent_src = fvnn.VDBTensor(noise.grid, latent_known)
            latent_src = vdb_utils.vdb_to_spconv(latent_src)

        grid, kmap = noise.grid, noise.kmap
        noise = vdb_utils.vdb_to_spconv(noise)
        if self.zero_noise:
            noise = noise.replace(torch.zeros_like(noise.feats))

        scheduler = copy.deepcopy(self.scheduler)
        timesteps, _ = retrieve_timesteps(
            scheduler, self.num_inference_steps, device, None
        )

        xyzs_list = [noise[i].xyzs for i in range(noise.shape[0])]
        if cfg:
            xyzs_list = [*xyzs_list, *xyzs_list]
            object_names_ = [[" "] for _ in range(B)] + object_names
            object_bboxes_ = [
                torch.zeros((1, 2, 3), device=device) for _ in range(B)
            ] + object_bboxes
        else:
            object_names_ = object_names
            object_bboxes_ = object_bboxes

        ins_mask = get_instance_mask(xyzs_list, object_bboxes_)

        context = self.conditioner(
            torch.cat(xyzs_list),
            object_names_,
            ins_mask=ins_mask,
            object_bboxes=object_bboxes_,
        )

        if cfg:
            if latent_src is not None:
                latent_src = sp.sparse_cat([latent_src, latent_src])

        controls = None
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        with self.ema_scope():
            latent = noise
            for t in tqdm(timesteps, desc=f"[R{rank}] Denoising", leave=False):
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.repeat(B)  # [B]

                if cfg:
                    timestep = timestep.repeat(2)
                    model_input = sp.sparse_cat([latent, latent])
                else:
                    model_input = latent

                if controlnet is not None:
                    controls = controlnet(
                        latent_src,
                        timestep,
                        xt=model_input,
                        embed=context,
                        return_intermediates=True,
                    )

                noise_pred = self.model(
                    model_input, timestep, embed=context, controls=controls
                ).feats

                # perform guidance
                if cfg:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (
                        noise_pred_text - noise_pred_uncond
                    )

                # compute the previous noisy sample x_t -> x_t-1
                latent = latent.replace(
                    scheduler.step(noise_pred, t, latent.feats, return_dict=False)[0]
                )

        return vdb_utils.spconv_to_vdb(latent, grid, kmap)

    @torch.no_grad()
    def multi_sample(
        self,
        noise: fvnn.VDBTensor,
        scene_latent_shape: Tuple[int, int, int],
        patch_latent_shape: List[int],
        patch_bbox_worlds: List[torch.Tensor],
        patch_object_names: List[List[str]],
        patch_object_bboxes: List[List[torch.Tensor]],
        bbox_world: Optional[torch.Tensor] = None,
        guidance_scale: float = 1.0,
        max_batch: int = 4,
        latent_src_patches: Optional[fvnn.VDBTensor] = None,
        controlnet: Optional[nn.Module] = None,
        cpu_offload: bool = False,
    ) -> fvnn.VDBTensor:
        """
        Generate a large 3D scene using MultiDiffusion:
        at each denoising step, split into overlapping latent patches,
        denoise in parallel, then fuse by averaging overlapping regions.

        Args:
            noise: The initial noisy latent scene.
            scene_latent_shape: Full scene latent shape [Lx, Ly, Lz].
            patch_latent_shape: Shape of each latent patch [Lx, Ly, Lz].
            patch_bbox_worlds: List of bounding boxes for each patch in world coords.
            patch_object_names: List of object names per patch.
            patch_object_bboxes: List of object bounding boxes per patch.
            bbox_world: Optional scene bounding box in world coordinates.
            guidance_scale: Classifier-free guidance scale.
            max_batch: Maximum number of patches to process in parallel.

        Returns:
            fvnn.VDBTensor: The denoised latent scene.
        """
        device = noise.device
        cfg = self.cond_drop_prob > 0 and guidance_scale != 1.0

        scheduler = copy.deepcopy(self.scheduler)
        timesteps, _ = retrieve_timesteps(
            scheduler, self.num_inference_steps, device, None
        )

        num_patches = len(patch_bbox_worlds)

        latent = noise
        # Build progress bar description with rank and scene size in meters
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        desc = f"[R{rank}] MultiDiffusion"
        if bbox_world is not None:
            scene_size = (
                bbox_world[0, 3:] - bbox_world[0, :3]
            )  # max - min, bbox is [B, 6]
            desc += f" ({scene_size[0]:.1f}x{scene_size[1]:.1f}x{scene_size[2]:.1f}m, {num_patches}p)"
        with self.ema_scope():
            for t in tqdm(timesteps, desc=desc, leave=False):
                # Split current latent into overlapping patches
                patches_vdb, _ = split_fvdb(
                    vdbtensor=latent,
                    patch_shape=patch_latent_shape,
                    patch_bbox_worlds=patch_bbox_worlds,
                    scene_bbox_world=bbox_world,
                )

                if latent_src_patches is not None:
                    src_filled = patches_vdb.grid.fill_from_grid(
                        latent_src_patches.data, latent_src_patches.grid, 0.0
                    )
                    src_sp = vdb_utils.vdb_to_spconv(
                        fvnn.VDBTensor(patches_vdb.grid, src_filled)
                    )

                patches_sp = vdb_utils.vdb_to_spconv(patches_vdb)

                # Process patches in mini-batches
                all_noise_preds = []
                chunk_src = None
                for chunk_start in range(0, num_patches, max_batch):
                    chunk_end = min(chunk_start + max_batch, num_patches)
                    chunk_size = chunk_end - chunk_start

                    chunk_sp = patches_sp[chunk_start:chunk_end]
                    timestep = t.repeat(chunk_size)

                    chunk_names = patch_object_names[chunk_start:chunk_end]
                    chunk_bboxes = patch_object_bboxes[chunk_start:chunk_end]

                    # Prepare inputs for classifier-free guidance
                    if cfg:
                        timestep = timestep.repeat(2)
                        model_input = sp.sparse_cat([chunk_sp, chunk_sp])
                        object_names_ = [[" "] for _ in range(chunk_size)] + chunk_names
                        object_bboxes_ = [
                            torch.zeros((1, 2, 3), device=device)
                            for _ in range(chunk_size)
                        ] + chunk_bboxes
                        if latent_src_patches is not None:
                            chunk_src = src_sp[chunk_start:chunk_end]
                            chunk_src = sp.sparse_cat([chunk_src, chunk_src])
                    else:
                        model_input = chunk_sp
                        object_names_ = chunk_names
                        object_bboxes_ = chunk_bboxes
                        if latent_src_patches is not None:
                            chunk_src = src_sp[chunk_start:chunk_end]

                    xyzs_list = [
                        model_input[i].xyzs for i in range(model_input.shape[0])
                    ]
                    ins_mask = get_instance_mask(xyzs_list, object_bboxes_)

                    context = self.conditioner(
                        torch.cat(xyzs_list),
                        object_names_,
                        ins_mask=ins_mask,
                        object_bboxes=object_bboxes_,
                    )

                    controls = None
                    if controlnet is not None:
                        controls = controlnet(
                            chunk_src,
                            timestep,
                            xt=model_input,
                            embed=context,
                            return_intermediates=True,
                        )

                    # Run model on chunk
                    chunk_pred = self.model(
                        model_input, timestep, embed=context, controls=controls
                    ).feats

                    # Apply classifier-free guidance
                    if cfg:
                        pred_uncond, pred_cond = chunk_pred.chunk(2)
                        chunk_pred = pred_uncond + guidance_scale * (
                            pred_cond - pred_uncond
                        )

                    if cpu_offload:
                        chunk_pred = chunk_pred.cpu()
                    all_noise_preds.append(chunk_pred)

                noise_pred = torch.cat(all_noise_preds, dim=0)
                if cpu_offload:
                    noise_pred = noise_pred.to(device)
                del all_noise_preds

                # MultiDiffusion: denoise per patch, then merge overlaps
                denoised_feats = scheduler.step(
                    noise_pred, t, patches_vdb.data.jdata, return_dict=False
                )[0]
                denoised_vdb = fvnn.VDBTensor(
                    patches_vdb.grid, patches_vdb.grid.jagged_like(denoised_feats)
                )
                latent = merge_fvdb(
                    patches=denoised_vdb,
                    patch_bbox_worlds=patch_bbox_worlds,
                    scene_latent_shape=scene_latent_shape,
                    cpu_offload=cpu_offload,
                )

        return latent
