import torch
import os
import copy
import math
import warnings
from skimage import io
import numpy as np
from typing import *
import fvdb.nn as fvnn

# Suppress FutureWarning from PyTorch's checkpoint.py about deprecated autocast API
# This is from PyTorch internals, not our code, and will be fixed in future PyTorch versions
warnings.filterwarnings("ignore", message=".*torch.cpu.amp.autocast.*is deprecated.*")
warnings.filterwarnings("ignore", message=".*torch.cuda.amp.autocast.*is deprecated.*")

from seen2scene.configs.opt import Generator
from seen2scene.tools import vis_utils, common_utils, vdb_utils, gif_utils
from seen2scene.tools.log_utils import get_logger
from seen2scene.models.fields import (
    Patch,
    TSDFPatch,
    split_kwargs,
    split_fvdb
)
from seen2scene.models.trainer.common import (
    sparsify_grid,
    Normalizer,
    get_instance_mask,
    cosine_schedule,
)
from .base import Net as BaseTrainer

logger = get_logger(file_name=__file__, debug="train_gen")


class Net(BaseTrainer):
    """PyTorch Lightning module for training flow matching generators on 3D scenes.

    Main training module that orchestrates:
    - Loading and freezing a pretrained VAE for latent space encoding
    - Training a flow matching diffusion model in the latent space
    - Computing losses, metrics, and visualizations
    - Handling distributed training and checkpointing

    The model operates in latent space: VAE encodes scenes to latents, generator
    learns to denoise latents, VAE decodes back to 3D representations.

    Attributes:
        vae: Dictionary containing frozen VAE model and normalizer.
        geo_model: Flow matching generator (SparseDiT or similar).
        metrics: Dictionary of metric computation modules.
        task: Task configuration (training, validation, or generation).
    """

    def __init__(self, args: Generator):
        """Initialize the generator training module.

        Loads a frozen pretrained VAE and instantiates the flow matching generator.
        Optionally loads generator weights from a checkpoint.

        Args:
            args: Generator configuration containing:
                - vae_cfg: VAE model configuration
                - vae_ckpt: Path to pretrained VAE checkpoint
                - model: Flow matching model configuration
                - ckpt_path: Optional generator checkpoint to resume from
                - task: Task configuration (train/val/generation)
                - learning_rate: Optimizer learning rate
                - Various other training parameters

        Notes:
            - VAE is loaded in eval mode and frozen (no gradients)
            - Generator can optionally resume from checkpoint with ignore_keys
            - Supports distributed training with DDP
        """
        super().__init__(args)
        self.ema_update_batch = args.ema_update_batch
        self.progressive_end = args.progressive_end
        self.sample_posterior = args.sample_posterior
        self.src_key = getattr(args, "src_key", None)
        self.avg_over = args.avg_over

        kwargs = {"map_location": lambda storage, loc: storage, "weights_only": True}

        # VAE ------------------------------------------------------------------
        self.vae = {
            "model": common_utils.instantiate_from_config(args.vae_cfg),
            "norm": Normalizer(args.vae_cfg["channels"], norm_by=args.norm_by),
        }
        ckpt_dict = torch.load(args.vae_ckpt, **kwargs)["state_dict"]
        for key in self.vae.keys():
            self.vae[key].load_state_dict(
                {
                    k.replace(f"{key}.", ""): v
                    for k, v in ckpt_dict.items()
                    if f"{key}." in k
                },
                strict=False,
            )
            self.vae[key].requires_grad_(False)
            self.vae[key].eval()
        logger.info(f"Resume VAE from {args.vae_ckpt}")

        # Generator ------------------------------------------------------------
        self.model = common_utils.instantiate_from_config(args.model)
        if args.ckpt_path is not None:
            state_dict = torch.load(args.ckpt_path, **kwargs)["state_dict"]
            if args.ignore_keys is not None:
                state_dict = {
                    k: v
                    for k, v in state_dict.items()
                    if not any(ignore_key in k for ignore_key in args.ignore_keys)
                }

            missing_keys, unexpected_keys = self.load_state_dict(
                state_dict, strict=False
            )
            logger.info(
                f"Resume from {args.ckpt_path}: missing_keys: {missing_keys}, unexpected_keys: {unexpected_keys}"
            )

    def setup(self, stage: str) -> None:
        """Setup models and move to correct devices for training/validation/testing.

        Called by PyTorch Lightning after model initialization. Moves all submodules
        to the appropriate device and sets up distributed training if needed.

        Args:
            stage: Training stage - "fit", "validate", "test", or "predict".

        Notes:
            - Moves VAE, metrics, and CLIP models to self.device
            - Ensures all frozen models (VAE, metrics) are in eval mode
        """
        super().setup(stage)
        self.metrics["model"].to(self.device)
        for key in self.vae.keys():
            self.vae[key].to(self.device)

    @torch.no_grad()
    def geometry_encode(
        self, fields: TSDFPatch, keep_region: Optional[List[str]] = None
    ) -> Tuple[fvnn.VDBTensor, Dict[str, Any]]:
        latent, band_mask, _, stats_dict = self.vae["model"].encode(
            fields,
            sample_posterior=self.sample_posterior,
            mask_unknown=self.mask_unknown,
            keep_region=keep_region,
        )
        if "norm" in self.vae:
            jdata = self.vae["norm"].normalize(latent.data.jdata)
            latent = fvnn.VDBTensor(latent.grid, latent.grid.jagged_like(jdata))
        return latent, band_mask, stats_dict

    @torch.no_grad()
    def geometry_decode(
        self,
        latent: fvnn.VDBTensor,
        field_params: Dict[str, Any],
        return_recon: bool = False,
    ) -> TSDFPatch:
        if "norm" in self.vae:
            jdata = self.vae["norm"].denormalize(latent.data.jdata)
            latent = fvnn.VDBTensor(latent.grid, latent.grid.jagged_like(jdata))
        recon_field = self.vae["model"].decode(
            latent, field_params=field_params, return_recon=return_recon
        )[0]["recon_field"]
        return recon_field

    def losses(self, batch, ema: bool = False) -> Tuple[Dict, Dict, Dict]:
        loss_dict, stats_dict = {}, {}

        prefix = "train" if self.training else "val"
        field_tgt = batch[self.latent_key]

        object_bboxes = field_tgt.object_bboxes
        object_names = field_tgt.object_names

        latent_tgt, band_mask = self.geometry_encode(
            field_tgt, keep_region=batch["keep_region"]
        )[:2]
        xyzs = latent_tgt.grid.grid_to_world(latent_tgt.grid.ijk.float())  # [N, 3]
        xyzs_list = [xyzs_i.jdata for xyzs_i in xyzs]
        ins_mask = get_instance_mask(xyzs_list, object_bboxes)
        stats_dict |= {
            f"{prefix}/latent-mean": latent_tgt.data.jdata.mean().item(),
            f"{prefix}/latent-std": latent_tgt.data.jdata.std().item(),
        }

        # Fix: Use grid length for VDBTensor batch size, not shape[0]
        batch_size = len(latent_tgt.grid)
        if len(object_bboxes) != batch_size:
            raise ValueError(
                f"Batch size mismatch: object_bboxes has {len(object_bboxes)} entries "
                f"but latent_tgt has batch_size={batch_size}"
            )

        if self.training and self.current_epoch < self.progressive_end:
            sparity_ratio = cosine_schedule(self.current_epoch, self.progressive_end)
            latent_tgt, keep_mask = sparsify_grid(
                latent_tgt, sparity_ratio, ins_mask=ins_mask
            )
            band_mask = band_mask[keep_mask]
            ins_mask = ins_mask[keep_mask]
            stats_dict |= {
                f"{prefix}/sparity-ratio": sparity_ratio,
                f"{prefix}/drop-ratio": 1 - keep_mask.float().mean().item(),
                f"{prefix}/band-ratio": band_mask.float().mean().item(),
            }

        # Monitor semantic class distribution
        if self.training:
            self.log_semantic_distribution(ins_mask, object_names, latent_tgt)

        kwargs = {
            "latent_tgt": latent_tgt,
            "band_mask": band_mask,
            "ins_mask": ins_mask,
            "object_bboxes": object_bboxes,
            "object_names": object_names,
            "avg_over": self.avg_over,
            "ema": ema,
        }
        if self.src_key is not None:
            latent_src = self.geometry_encode(batch[self.src_key])[0]
            kwargs["latent_src"] = latent_src

        loss_geometry, loss_stats = self.model.losses(**kwargs)

        loss_dict |= {
            f"{prefix}/loss-geometry-{field_tgt.voxel_size:.3f}": loss_geometry
        }

        return loss_dict, stats_dict

    def on_train_epoch_start(self):
        if hasattr(self.model, "maybe_reload_base"):
            self.model.maybe_reload_base()

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if getattr(self.model, "use_ema", False) and (
            (batch_idx + 1) % self.ema_update_batch == 0
        ):
            self.model.model_ema(self.model.model)

    @torch.no_grad()
    def generate_patch(self, batch, return_latent: bool = False) -> TSDFPatch:
        patch_shape = [256, 256, 256]

        # When drop_bbox is enabled, replace bboxes/names with unconditional placeholders
        patch_kwargs = copy.deepcopy(batch[self.latent_key].kwargs)
        drop_bbox = getattr(self.task, "drop_bbox", False)
        if drop_bbox:
            batch_size = len(batch["scene_names"])
            patch_kwargs["object_bboxes"] = [
                torch.zeros((1, 2, 3), device=self.device) for _ in range(batch_size)
            ]
            patch_kwargs["object_names"] = [[" "] for _ in range(batch_size)]
        patch = Patch(patch_kwargs)

        noise, scene_latent_shape = self.randn_like(patch)
        guidance_scale = 1.0 if drop_bbox else self.task.guidance_scale
        kwargs = {"noise": noise, "guidance_scale": guidance_scale}

        if (
            self.task.scene_shape is not None
            and list(self.task.scene_shape) == patch_shape
        ):
            if self.src_key is not None:
                kwargs["latent_src"] = self.geometry_encode(batch[self.src_key])[0]
            kwargs |= {
                "object_names": patch.object_names,
                "object_bboxes": patch.object_bboxes,
            }
            latent = self.model.sample(**kwargs)
        else:
            cpu_offload = getattr(self.task, "cpu_offload", False)
            patch_latent_shape = [x // self.vae["model"].factor for x in patch_shape]

            kwargs_patches = split_kwargs(
                patch.kwargs,
                patch_shape=patch_shape,
                overlap=self.task.overlap,
                factor=self.vae["model"].factor,
            )
            num_patches = len(kwargs_patches["bbox_world"])

            # Override patch-level bboxes/names when drop_bbox is enabled
            if drop_bbox:
                kwargs_patches["object_bboxes"] = [
                    torch.zeros((1, 2, 3), device=patch.device)
                    for _ in range(num_patches)
                ]
                kwargs_patches["object_names"] = [[" "] for _ in range(num_patches)]
            if self.src_key is not None:
                latent_src = self.geometry_encode(batch[self.src_key])[0]
                latent_src_patches, _ = split_fvdb(
                    vdbtensor=latent_src,
                    patch_shape=patch_latent_shape,
                    patch_bbox_worlds=kwargs_patches["bbox_world"],
                    scene_bbox_world=patch.bbox_world,
                    cpu_offload=cpu_offload,
                )
                kwargs["latent_src_patches"] = latent_src_patches

            latent = self.model.multi_sample(
                scene_latent_shape=scene_latent_shape,
                patch_latent_shape=patch_latent_shape,
                patch_bbox_worlds=kwargs_patches["bbox_world"],
                patch_object_names=kwargs_patches["object_names"],
                patch_object_bboxes=kwargs_patches["object_bboxes"],
                bbox_world=patch.bbox_world,
                cpu_offload=cpu_offload,
                **kwargs,
            )

        patch = self.geometry_decode(
            latent, field_params=patch.kwargs, return_recon=True
        )

        if return_latent:
            return patch, latent
        return patch

    def randn_like(self, patch: Patch) -> Tuple[fvnn.VDBTensor, List[int]]:
        latent_shape = [y // self.vae["model"].factor for y in patch.patch_shape]
        voxel_size = patch.voxel_size * self.vae["model"].factor

        noise = torch.randn(
            (patch.batch_size, *latent_shape, self.vae["model"].channels),
            device=patch.device,
        )
        noise = fvnn.vdbtensor_from_dense(
            noise, voxel_sizes=[voxel_size] * 3, origins=[voxel_size / 2.0] * 3
        )
        return noise, latent_shape

    @torch.no_grad()
    def render_outputs(
        self, batch, field_pred: Union[TSDFPatch, List[TSDFPatch]], **render_kwargs
    ) -> OrderedDict:
        src_dict, gen_dict, tgt_dict = OrderedDict(), OrderedDict(), OrderedDict()

        if self.src_key is not None and isinstance(batch[self.src_key], TSDFPatch):
            src_dict |= {
                f"{self.src_key} (Src)": batch[self.src_key].render_field(
                    **render_kwargs
                ),
                "structure (Src)": batch[self.src_key].render_structure(
                    **render_kwargs
                ),
                "recon (Src)": self.vae["model"](
                    batch[self.src_key],
                    mask_unknown=self.mask_unknown,
                    sample_posterior=self.sample_posterior,
                    return_recon=True,
                )[-1]["recon_field"].render_field(**render_kwargs),
            }

        for idx, field_i in enumerate(field_pred):
            if isinstance(field_i, TSDFPatch):
                gen_dict |= {
                    f"tsdf (Generation)_{idx}": field_i.render_field(**render_kwargs),
                    f"structure (Generation)_{idx}": field_i.render_structure(
                        **render_kwargs
                    ),
                }

        tgt_dict |= {
            f"Bbox (Target)": batch[self.latent_key].plot_bounding_boxes(
                **render_kwargs
            )
        }
        if isinstance(batch[self.latent_key], TSDFPatch):
            tgt_dict |= {
                f"{self.latent_key} (Target)": batch[self.latent_key].render_field(
                    **render_kwargs
                ),
                "structure (Target)": batch[self.latent_key].render_structure(
                    **render_kwargs
                ),
                f"tsdf (Target Recon)": self.vae["model"](
                    batch[self.latent_key],
                    mask_unknown=self.mask_unknown,
                    sample_posterior=self.sample_posterior,
                    return_recon=True,
                )[-1]["recon_field"].render_field(**render_kwargs),
            }

        if "mesh_gt" in batch:
            tgt_dict |= {
                "mesh (Ground Truth)": batch["mesh_gt"].render_field(**render_kwargs)
            }

        return src_dict | gen_dict | tgt_dict

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        batch_size = len(batch["scene_names"])
        log_kwargs = {
            "prog_bar": False,
            "logger": True,
            "on_step": False,
            "on_epoch": True,
            "batch_size": batch_size,
            "sync_dist": True,
            "rank_zero_only": True,
        }

        loss_dict_no_ema, stats_dict_no_ema = self.losses(batch)
        self.log_dict(loss_dict_no_ema, **log_kwargs)
        self.log_dict(stats_dict_no_ema, **log_kwargs)

        # self.model_ema
        if getattr(self.model, "use_ema", False):
            loss_dict_ema = self.losses(batch, ema=True)[0]
            loss_dict_ema = {key + "_ema": loss_dict_ema[key] for key in loss_dict_ema}
            self.log_dict(loss_dict_ema, **log_kwargs)

        world_size = self.trainer.world_size
        rank = self.trainer.global_rank
        global_batch_idx = batch_idx * world_size + rank
        global_sample_idx = global_batch_idx * batch_size
        if global_sample_idx >= self.task.num_samples:
            return

        fields_gen = [self.generate_patch(batch)]
        for _ in range(max(0, self.task.num_variations - 1)):
            fields_gen.append(self.generate_patch(batch))

        mesh_field_tgt = (
            batch["mesh_gt"] if "mesh_gt" in batch else batch[self.latent_key]
        )
        self.metrics["model"].update(
            field_pred=fields_gen,
            volume_field_tgt=batch[self.latent_key],
            mesh_field_tgt=mesh_field_tgt,
            num_views_for_fid=self.task.num_views_for_fid,
        )
        if batch_idx == 0:
            images_dict = self.render_outputs(
                batch,
                fields_gen,
                resolution=self.task.img_wh,
                num_views=1,
                backend=self.task.backend,
            )
            images, titles = vis_utils.batch2row_types2column(images_dict)
            image_grid = vis_utils.make_image_grid(images, titles)
            if self.trainer.is_global_zero:
                self.logger.log_image(
                    f"Validation_batch_{batch_idx}", [image_grid], step=self.global_step
                )

    @torch.no_grad()
    def export_field(self, field: Patch, field_name: str, scene_names):
        field.export_objects_as_json(self.log_dir, scene_names)
        field.export_objects_as_mesh(self.log_dir, scene_names)

        if "mesh" in self.task.export_as and hasattr(field, "export_mesh"):
            field.export_mesh(self.log_dir, scene_names, postfix=field_name)

        if "bbox" in self.task.export_as and hasattr(field, "export_objects_as_mesh"):
            field.export_objects_as_mesh(self.log_dir, scene_names, postfix=field_name)

        if "npz" in self.task.export_as and hasattr(field, "export_scenes_as_npz"):
            field.export_scenes_as_npz(self.log_dir, scene_names, postfix=field_name)

        if "volume" in self.task.export_as and hasattr(field, "export_volumes"):
            field.export_volumes(self.log_dir, scene_names, postfix=field_name)

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        batch_size = len(batch["scene_names"])
        world_size = self.trainer.world_size
        rank = self.trainer.global_rank
        global_batch_idx = batch_idx * world_size + rank
        global_sample_idx = global_batch_idx * batch_size

        if global_sample_idx >= self.task.num_samples:
            return

        # Show scene name in progress bar
        scene_label = batch["scene_names"][0]
        pbar = getattr(self.trainer, "progress_bar_callback", None)
        if pbar is not None and hasattr(pbar, "test_progress_bar"):
            pbar.test_progress_bar.set_postfix_str(scene_label, refresh=True)

        # Skip existing samples when overwrite is disabled
        if not self.task.overwrite:
            all_exist = True
            for scene_name, bbox_world_i in zip(
                batch["scene_names"], batch[self.latent_key].bbox_world
            ):
                bbox_str = common_utils.bbox2str(bbox_world_i)
                out_dir_i = os.path.join(self.log_dir, f"{scene_name}_{bbox_str}")
                mesh_path = os.path.join(out_dir_i, "mesh_tsdf (Generation)_0.ply")
                if not os.path.exists(mesh_path):
                    all_exist = False
                    break
            if all_exist:
                self.log("skip", float(batch_size), prog_bar=True, reduce_fx="sum")
                return

        # Export PCA-colored latent point clouds
        if getattr(self.task, "vis_latent_pca", False):
            from seen2scene.tools.vdb_utils import export_latent_pca

            export_latent_pca(
                latent_vdb=latent_gen,
                bbox_world=batch[self.latent_key].bbox_world,
                scene_names=batch["scene_names"],
                out_dir=self.log_dir,
            )

        # Export RoPE PCA visualizations (CLIP embeddings with distance conditioning)
        if getattr(self.task, "vis_rope_pca", False):
            # Get target field
            tgt_field = batch[self.latent_key]
            # Encode the target field to get latent representation
            latent_tgt, band_mask = self.geometry_encode(
                tgt_field, keep_region=batch["keep_region"]
            )[:2]

            xyzs = latent_tgt.grid.grid_to_world(latent_tgt.grid.ijk.float())  # [N, 3]
            xyzs_list = [xyzs_i.jdata for xyzs_i in xyzs]
            ins_mask = get_instance_mask(xyzs_list, tgt_field.object_bboxes)
            latent_sp = vdb_utils.vdb_to_spconv(latent_tgt)
            # Use conditioner to get CLIP embeddings with RoPE (just like in training)
            # This directly calls the conditioning path used during training
            with torch.no_grad():
                # Conditioner expects plain world-space coordinate tensor, not SparseTensor
                world_coords = torch.cat(xyzs_list)
                # WITH distance RoPE
                context_with_rope = self.model.conditioner(
                    world_coords,
                    tgt_field.object_names,
                    ins_mask=ins_mask,
                    object_bboxes=tgt_field.object_bboxes,  # Enables distance RoPE
                )

                # WITHOUT distance RoPE (pass None for object_bboxes)
                context_without_rope = self.model.conditioner(
                    world_coords,
                    tgt_field.object_names,
                    ins_mask=ins_mask,
                    object_bboxes=None,  # Disables distance RoPE
                )

            vdb_utils.export_rope_pca_visualizations(
                latent_sp=latent_sp,
                context_with_rope=context_with_rope,
                context_without_rope=context_without_rope,
                scene_names=batch["scene_names"],
                out_dir=self.log_dir,
                object_names_batch=tgt_field.object_names,
                object_bboxes_batch=tgt_field.object_bboxes,
                ins_mask=ins_mask,
                bbox_world=tgt_field.bbox_world,
            )

        if getattr(self.task, "vis_sparity", False):
            field_tgt = batch[self.latent_key]
            latent_tgt = self.geometry_encode(
                field_tgt, keep_region=batch["keep_region"]
            )[0]
            xyzs = latent_tgt.grid.grid_to_world(latent_tgt.grid.ijk.float())  # [N, 3]
            xyzs_list = [xyzs_i.jdata for xyzs_i in xyzs]  # List of [N_i, 3] tensors
            ins_mask = get_instance_mask(xyzs_list, field_tgt.object_bboxes)
            for sparity_ratio in [0.0, 0.01, 0.1, 0.5]:
                latent_i, _ = sparsify_grid(
                    latent_tgt, sparity_ratio=sparity_ratio, ins_mask=ins_mask
                )
                vdb_utils.vis_grid_as_pointclouds(
                    latent_i.grid,
                    field_tgt.bbox_world,
                    batch["scene_names"],
                    self.log_dir,
                    sparity_ratio,
                )

        if getattr(self.task, "vis_scan_prob", False):
            from einops import rearrange
            from seen2scene.configs.dataset import Voxel

            field_tgt = batch[self.latent_key]
            factor = self.vae["model"].factor
            latent_tgt = self.geometry_encode(
                field_tgt, keep_region=batch["keep_region"]
            )[0]

            # Compute struct_down (same as encoder.structure_encoding)
            struct = field_tgt.structure  # [B, 1, D, H, W] int8
            struct_down = torch.nn.functional.max_pool3d(
                struct.to(torch.float16), kernel_size=factor, stride=factor
            )
            struct_down = rearrange(struct_down, "b c h w d -> b h w d c").contiguous()
            struct_down = fvnn.vdbtensor_from_dense(
                struct_down,
                [0, 0, 0],
                voxel_sizes=latent_tgt.grid.voxel_sizes,
                origins=latent_tgt.grid.origins,
            )

            # Compute instance mask from struct_down grid (same as encoder.py)
            xyzs = struct_down.grid.grid_to_world(struct_down.grid.ijk.float())
            xyzs_list = [xyzs_i.jdata for xyzs_i in xyzs]
            ins_mask = get_instance_mask(xyzs_list, field_tgt.object_bboxes)

            vdb_utils.export_scan_prob_heatmap(
                struct_down,
                field_tgt.bbox_world,
                batch["scene_names"],
                self.log_dir,
                ins_mask=ins_mask,
            )

        self.log("gen", float(batch_size), prog_bar=True, reduce_fx="sum")

        # Layout editing: modify object layout via LLM before generation
        if self.task.name == "layout_edit" and self.task.prompt:
            from seen2scene.models.trainer.layout_edit_utils import apply_layout_edit

            apply_layout_edit(batch, self.latent_key, self.task.prompt, self.log_dir)

        fields_dict = {f"tsdf (Target)": batch[self.latent_key]}
        if self.src_key is not None:
            src_field_tree = batch[self.src_key].build_field_tree(
                int(math.log2(self.vae["model"].factor)) + 1
            )
            fields_dict |= {f"tsdf (Input)": src_field_tree[0]}
        if "mesh_gt" in batch:
            fields_dict |= {"mesh (Ground Truth)": batch["mesh_gt"]}

        if isinstance(batch[self.latent_key], TSDFPatch):
            fields_dict |= {
                "tsdf (VAE)": self.vae["model"](
                    batch[self.latent_key],
                    sample_posterior=self.sample_posterior,
                    mask_unknown=self.mask_unknown,
                    return_recon=True,
                )[-1]["recon_field"]
            }

        scene_names = batch["scene_names"]
        for key, field_i in fields_dict.items():
            self.export_field(field_i, key, scene_names)

        fields_gen = []
        for i in range(self.task.repeat_num):
            field_gen_i = self.generate_patch(batch)
            fields_gen.append(field_gen_i)
            self.export_field(field_gen_i, f"tsdf (Generation)_{i}", scene_names)

        # Export images ------------------------------------------------------------
        images_dict = self.render_outputs(
            batch,
            field_pred=fields_gen,
            resolution=self.task.img_wh,
            num_views=4,
            theta=60.0,
            light_intensity=6.0,
            backend=self.task.backend,
        )
        images, titles = vis_utils.views2row_types2column(images_dict)
        for image_grid, bbox_world_i, scene_name in zip(
            images, fields_gen[0].bbox_world, batch["scene_names"]
        ):
            bbox_str = common_utils.bbox2str(bbox_world_i)
            out_dir = os.path.join(self.log_dir, f"{scene_name}_{bbox_str}")
            os.makedirs(out_dir, exist_ok=True)

            image_grid = vis_utils.make_image_grid(
                image_grid, titles, nrow=1, font_size=16
            )
            img_np = image_grid.permute(1, 2, 0).cpu().numpy()
            img_np = (img_np * 255).astype(np.uint8)  # Convert to uint8
            io.imsave(os.path.join(out_dir, "image_grid.png"), img_np)

        # When multiple repetitions were generated, create a GIF where each
        # frame shows one generation variation alongside the shared src/tgt.
        if self.task.repeat_num > 1:
            src_keys = [k for k in images_dict if "(Src)" in k]
            tgt_keys = [
                k for k in images_dict if "(Target)" in k or "Ground Truth" in k
            ]
            gif_frames_per_scene: List[List[np.ndarray]] = [
                [] for _ in batch["scene_names"]
            ]
            for rep_idx in range(self.task.repeat_num):
                gen_keys = [k for k in images_dict if f"(Generation)_{rep_idx}" in k]
                frame_dict = OrderedDict(
                    (k, images_dict[k]) for k in src_keys + gen_keys + tgt_keys
                )
                frame_images, frame_titles = vis_utils.views2row_types2column(
                    frame_dict
                )
                for scene_idx, frame_i in enumerate(frame_images):
                    frame_grid = vis_utils.make_image_grid(
                        frame_i, frame_titles, nrow=1, font_size=16
                    )
                    frame_np = frame_grid.permute(1, 2, 0).cpu().numpy()
                    frame_np = (frame_np * 255).astype(np.uint8)
                    gif_frames_per_scene[scene_idx].append(frame_np)

            for frames, bbox_world_i, scene_name in zip(
                gif_frames_per_scene, fields_gen[0].bbox_world, batch["scene_names"]
            ):
                bbox_str = common_utils.bbox2str(bbox_world_i)
                out_dir = os.path.join(self.log_dir, f"{scene_name}_{bbox_str}")
                os.makedirs(out_dir, exist_ok=True)
                gif_utils.save_gif(
                    frames,
                    os.path.join(out_dir, "generation_variations.gif"),
                    fps=1.0,
                )
