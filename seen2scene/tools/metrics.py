import torchmetrics as tm
from typing import *
import sys
import os
import json
import torch
import torch.nn as nn
import trimesh
import numpy as np
from functools import partial

from .mesh_utils import mesh2pclouds
from .log_utils import get_logger

logger = get_logger(file_name=__file__)


def _validate_mesh(mesh: trimesh.Trimesh, label: str = "mesh") -> bool:
    """Check that a mesh is non-degenerate and has finite vertices.

    Returns True if the mesh is valid, False otherwise.
    """
    if mesh is None:
        logger.warning(f"{label}: mesh is None")
        return False
    if not hasattr(mesh, "vertices") or not hasattr(mesh, "faces"):
        logger.warning(f"{label}: missing vertices/faces attributes")
        return False
    if len(mesh.vertices) == 0 or len(mesh.faces) == 0:
        logger.warning(f"{label}: empty mesh (verts={len(mesh.vertices)}, faces={len(mesh.faces)})")
        return False
    if not np.all(np.isfinite(mesh.vertices)):
        logger.warning(f"{label}: mesh has NaN/inf in vertices")
        return False
    return True


def _mesh2voxels(
    meshes: List[trimesh.Trimesh],
    bbox_world: np.ndarray,
    voxel_size: float,
    num_samples: int = 500_000,
) -> torch.Tensor:
    """Surface-voxelize meshes within bounding boxes via point sampling.

    Uses trimesh surface sampling instead of Open3D to avoid C-extension
    segfaults on meshes with degenerate geometry or NaN vertices.

    Args:
        meshes: List of trimesh.Trimesh objects.
        bbox_world: [B, 6] array of [xmin, ymin, zmin, xmax, ymax, zmax].
        voxel_size: Voxel edge length in world units.
        num_samples: Number of surface sample points per mesh.

    Returns:
        Boolean tensor of shape [B, G1, G2, G3].
    """
    voxelgrids = []
    for i, mesh in enumerate(meshes):
        bbox_min = bbox_world[i][:3].astype(np.float64)
        bbox_max = bbox_world[i][3:].astype(np.float64)
        grid_dims = np.round((bbox_max - bbox_min) / voxel_size).astype(int)
        grid_dims = np.maximum(grid_dims, 1)
        grid = np.zeros(grid_dims, dtype=bool)

        if _validate_mesh(mesh, label=f"_mesh2voxels[{i}]"):
            # Sample points on the mesh surface (pure barycentric — no C extensions)
            points, _ = trimesh.sample.sample_surface(mesh, num_samples)

            # Skip if sampling produced NaN (degenerate mesh)
            if not np.all(np.isfinite(points)):
                logger.warning(f"_mesh2voxels[{i}]: NaN/inf in sampled points, skipping")
                voxelgrids.append(torch.from_numpy(grid))
                continue

            # Convert world coords → voxel indices
            ijk = np.floor((points - bbox_min) / voxel_size).astype(int)

            # Keep only points inside the bounding box
            valid = np.all((ijk >= 0) & (ijk < grid_dims), axis=1)
            ijk = ijk[valid]

            if len(ijk) > 0:
                grid[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = True

        voxelgrids.append(torch.from_numpy(grid))

    return torch.stack(voxelgrids)


class IoU:
    def __init__(self):
        self.values = []
        self.smooth = 1e-6

    def update(
        self,
        mesh_gen: Union[List[List[trimesh.Trimesh]], List[trimesh.Trimesh]],
        mesh_tgt: List[trimesh.Trimesh],
        bbox_world: np.ndarray,
        voxel_size: float,
        device: str = "cuda:0",
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            preds: Predicted meshes, shape [B, N, 3] * num_variations
            target: Target meshes, shape [B, N, 3]
            resolution: Resolution for voxelgrids
        """
        if not isinstance(mesh_gen[0], list):
            mesh_gen = [mesh_gen]

        num_variations = len(mesh_gen)

        # [B, G1, G2, G3]
        tgt_bin = _mesh2voxels(mesh_tgt, bbox_world, voxel_size).to(device)

        iou = []
        for i in range(num_variations):
            pred_bin = _mesh2voxels(mesh_gen[i], bbox_world, voxel_size).to(device)
            assert pred_bin.shape == tgt_bin.shape

            mean_dims = [i for i in range(1, pred_bin.ndim)]  # [1, 2, ...]
            intersection = (pred_bin & tgt_bin).float().sum(dim=mean_dims)  # [B]
            union = (pred_bin | tgt_bin).float().sum(dim=mean_dims)  # [B]
            iou_i = (intersection + self.smooth) / (union + self.smooth)  # [B]
            iou.append(iou_i)

        iou = torch.stack(iou, dim=-1)  # [B, K]
        self.values.append(iou)

        return {
            "iou_avg": iou.mean(dim=-1).tolist(),
            "iou_min": iou.amin(dim=-1).tolist(),
            "iou_max": iou.amax(dim=-1).tolist(),
        }

    def compute(self, key: str) -> torch.Tensor:
        if not self.values:
            return {}
        values = torch.cat(self.values)  # [B, K]
        return {
            f"{key}_avg": torch.mean(values),
            f"{key}_min": torch.mean(values.amin(dim=-1)),
            f"{key}_max": torch.mean(values.amax(dim=-1)),
        }


class FID(tm.image.fid.FrechetInceptionDistance):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def compute(self, key: str):
        return {key: super().compute()}


class VolumeMetric:
    def __init__(self, loss_type: str = "l2"):
        self.loss_type = loss_type
        self.values = []

    def update(
        self,
        volume_gen: Union[List[torch.Tensor], torch.Tensor],
        volume_gt: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            preds: Predicted latent, shape [B, C, G1, G2, G3] * num_variations
            target: Target latent, shape [B, C, G1, G2, G3]
            mask: Mask, shape [B, C, G1, G2, G3]
        """
        if isinstance(volume_gen, torch.Tensor):
            volume_gen = [volume_gen]

        error = []
        for i in range(len(volume_gen)):
            preds_i = volume_gen[i]
            assert preds_i.shape == volume_gt.shape
            from .loss_utils import masked_weighted_loss
            loss = masked_weighted_loss(
                preds_i,
                volume_gt,
                mask=mask,
                loss_type=self.loss_type,
                batch_reduction=None,
            )
            error.append(loss)

        error = torch.stack(error, dim=-1)  # [B, num_variations]
        self.values.append(error)

        return {
            f"{self.loss_type}_avg": error.mean(dim=-1).tolist(),
            f"{self.loss_type}_min": error.amin(dim=-1).tolist(),
            f"{self.loss_type}_max": error.amax(dim=-1).tolist(),
        }

    def compute(self, key: str) -> torch.Tensor:
        if not self.values:
            return {}
        values = torch.cat(self.values)  # [B, num_variations]
        return {
            f"{key}_avg": torch.mean(values),
            f"{key}_min": torch.mean(values.amin(dim=-1)),
            f"{key}_max": torch.mean(values.amax(dim=-1)),
        }


def _chamfer_one_direction(
    src: torch.Tensor, tgt: torch.Tensor, chunk_size: int = 2000
) -> torch.Tensor:
    """Mean nearest-neighbour squared-L2 from *src* to *tgt*.

    Args:
        src: [B, N, 3]
        tgt: [B, M, 3]
        chunk_size: Process src in chunks to limit GPU memory.

    Returns:
        [B] mean nearest-neighbour squared-L2 distance.
    """
    B, N, _ = src.shape
    min_dists = []
    for start in range(0, N, chunk_size):
        src_chunk = src[:, start : start + chunk_size]  # [B, chunk, 3]
        # [B, chunk, 1, 3] - [B, 1, M, 3] -> squared L2
        diff = src_chunk.unsqueeze(2) - tgt.unsqueeze(1)
        dist_sq = (diff**2).sum(dim=-1)  # [B, chunk, M]
        min_dists.append(dist_sq.amin(dim=-1))  # [B, chunk]
    min_dists = torch.cat(min_dists, dim=1)  # [B, N]
    return min_dists.mean(dim=-1)  # [B]


def _cd_between(pc_a: torch.Tensor, pc_b: torch.Tensor) -> torch.Tensor:
    """Bidirectional Chamfer Distance between two point clouds.

    Args:
        pc_a: [N, 3]
        pc_b: [M, 3]

    Returns:
        Scalar CD value.
    """
    a = pc_a.unsqueeze(0)  # [1, N, 3]
    b = pc_b.unsqueeze(0)  # [1, M, 3]
    return (
        _chamfer_one_direction(a, b).squeeze() + _chamfer_one_direction(b, a).squeeze()
    ) / 2.0


def _pairwise_cd_matrix(
    pcs_a: List[torch.Tensor],
    pcs_b: List[torch.Tensor],
    device: str = "cuda:0",
    batch_size: int = 64,
) -> torch.Tensor:
    """Pairwise Chamfer Distance matrix between two sets of point clouds.

    Args:
        pcs_a: List of [P, 3] tensors (N_a elements).
        pcs_b: List of [P, 3] tensors (N_b elements).
        device: Device for GPU-accelerated CD.
        batch_size: Not used (kept for API compatibility).

    Returns:
        [N_a, N_b] distance matrix.
    """
    N_a, N_b = len(pcs_a), len(pcs_b)
    dist = torch.zeros(N_a, N_b)
    for i in range(N_a):
        a = pcs_a[i].float().to(device)
        for j in range(N_b):
            b = pcs_b[j].float().to(device)
            dist[i, j] = _cd_between(a, b).cpu()
    return dist



def _cov_mmd(dist_matrix: torch.Tensor) -> Tuple[float, float]:
    """Compute Coverage and MMD from a [N_gen, N_ref] distance matrix.

    COV = fraction of reference shapes that are the nearest neighbour of
          at least one generated shape.
    MMD = mean distance from each reference to its nearest generated shape.

    Args:
        dist_matrix: [N_gen, N_ref] pairwise distances.

    Returns:
        (coverage, mmd) tuple.
    """
    N_gen, N_ref = dist_matrix.shape

    # For each gen shape, find its nearest ref → which refs are "covered"
    nearest_ref_per_gen = dist_matrix.argmin(dim=1)  # [N_gen]
    covered = nearest_ref_per_gen.unique().numel()
    cov = covered / N_ref

    # For each ref shape, find its nearest gen shape → MMD
    min_dist_per_ref = dist_matrix.amin(dim=0)  # [N_ref]
    mmd = min_dist_per_ref.mean().item()

    return cov, mmd


def _one_nna(
    M_rr: torch.Tensor, M_rs: torch.Tensor, M_ss: torch.Tensor
) -> float:
    """1-Nearest-Neighbor Accuracy (leave-one-out).

    Builds the full (N_r + N_s) x (N_r + N_s) distance matrix and performs
    leave-one-out 1-NN classification. Labels: ref=1, gen=0.

    Args:
        M_rr: [N_r, N_r] ref-ref distances.
        M_rs: [N_r, N_s] ref-gen distances.
        M_ss: [N_s, N_s] gen-gen distances.

    Returns:
        1-NNA accuracy in [0, 1]. Ideal generative model → 0.5.
    """
    N_r, N_s = M_rs.shape
    N = N_r + N_s

    # Build full distance matrix
    #   [ M_rr  M_rs ]
    #   [ M_sr  M_ss ]
    M_sr = M_rs.t()  # [N_s, N_r]
    D = torch.cat(
        [torch.cat([M_rr, M_rs], dim=1), torch.cat([M_sr, M_ss], dim=1)], dim=0
    )  # [N, N]

    # Leave-one-out: set diagonal to infinity
    D.fill_diagonal_(float("inf"))

    # 1-NN: find nearest neighbour index for each sample
    nn_idx = D.argmin(dim=1)  # [N]

    # Labels: first N_r are ref (1), last N_s are gen (0)
    labels = torch.cat([torch.ones(N_r), torch.zeros(N_s)])
    nn_labels = labels[nn_idx]

    # Accuracy: fraction correctly classified
    correct = (labels == nn_labels).float().sum()
    return (correct / N).item()


class ChamferDistance:
    """Bidirectional Chamfer Distance between generated and target meshes."""

    def __init__(self, num_points: int = 10000):
        self.values = []
        self.num_points = num_points

    def update(
        self,
        mesh_gen: Union[List[List[trimesh.Trimesh]], List[trimesh.Trimesh]],
        mesh_tgt: List[trimesh.Trimesh],
        device: str = "cuda:0",
    ) -> Dict[str, list]:
        if not isinstance(mesh_gen[0], list):
            mesh_gen = [mesh_gen]

        num_variations = len(mesh_gen)

        # [B, N, 3]
        tgt_pts = mesh2pclouds(mesh_tgt, num_points=self.num_points).float().to(device)

        cd = []
        for i in range(num_variations):
            pred_pts = (
                mesh2pclouds(mesh_gen[i], num_points=self.num_points).float().to(device)
            )
            dist_p2t = _chamfer_one_direction(pred_pts, tgt_pts)  # [B]
            dist_t2p = _chamfer_one_direction(tgt_pts, pred_pts)  # [B]
            cd.append((dist_p2t + dist_t2p) / 2.0)

        cd = torch.stack(cd, dim=-1)  # [B, K]
        self.values.append(cd)

        return {
            "chamfer_avg": cd.mean(dim=-1).tolist(),
            "chamfer_min": cd.amin(dim=-1).tolist(),
            "chamfer_max": cd.amax(dim=-1).tolist(),
        }

    def compute(self, key: str) -> Dict[str, torch.Tensor]:
        if not self.values:
            return {}
        values = torch.cat(self.values)  # [B, K]
        return {
            f"{key}_avg": torch.mean(values),
            f"{key}_min": torch.mean(values.amin(dim=-1)),
            f"{key}_max": torch.mean(values.amax(dim=-1)),
        }


class TMD:
    """Total Mutual Difference – diversity across K completions via pairwise CD."""

    def __init__(self, num_points: int = 10000):
        self.values = []
        self.num_points = num_points

    def update(
        self,
        mesh_gen: Union[List[List[trimesh.Trimesh]], List[trimesh.Trimesh]],
        device: str = "cuda:0",
    ) -> Dict[str, list]:
        if not isinstance(mesh_gen[0], list):
            mesh_gen = [mesh_gen]

        K = len(mesh_gen)
        if K < 2:
            logger.warning(f"TMD requires repeat_num > 1 (got K={K}); skipping.")
            return {}

        B = len(mesh_gen[0])

        # Sample points for every variation: list of [B, N, 3]
        all_pts = [
            mesh2pclouds(mesh_gen[k], num_points=self.num_points).float().to(device)
            for k in range(K)
        ]

        # Average pairwise CD across all K*(K-1)/2 pairs
        tmd = torch.zeros(B, device=device)
        num_pairs = 0
        for i in range(K):
            for j in range(i + 1, K):
                cd_ij = (
                    _chamfer_one_direction(all_pts[i], all_pts[j])
                    + _chamfer_one_direction(all_pts[j], all_pts[i])
                ) / 2.0
                tmd += cd_ij
                num_pairs += 1

        tmd = tmd / num_pairs  # [B]
        self.values.append(tmd.unsqueeze(-1))  # [B, 1]

        return {"tmd_avg": tmd.tolist()}

    def compute(self, key: str) -> Dict[str, torch.Tensor]:
        if not self.values:
            return {}
        values = torch.cat(self.values)  # [B, 1]
        return {f"{key}_avg": torch.mean(values)}


class GenerationMetrics:
    """Set-level generation metrics: COV, MMD, 1-NNA (CD-based).

    Unlike per-scene metrics, these compare the *entire* generated set against
    the *entire* reference set via pairwise distance matrices. Point clouds are
    accumulated across batches in update() and all 3 values are computed
    together in compute().

    Reference: Achlioptas et al. (ICML 2018), Yang et al. / PointFlow (ICCV 2019).
    """

    def __init__(self, num_points: int = 2048):
        self.num_points = num_points
        self.gen_pcs: List[torch.Tensor] = []  # list of [P, 3] tensors (CPU)
        self.ref_pcs: List[torch.Tensor] = []

    def update(
        self,
        mesh_gen: Union[List[List[trimesh.Trimesh]], List[trimesh.Trimesh]],
        mesh_ref: List[trimesh.Trimesh],
        device: str = "cuda:0",
    ) -> Dict:
        """Accumulate point clouds for later set-level computation.

        Args:
            mesh_gen: Generated meshes. If List[List[trimesh]], uses first
                      variation only (index 0).
            mesh_ref: Reference / target meshes.
            device: Unused (stored on CPU); kept for API consistency.

        Returns:
            Empty dict (no per-batch results for set-level metrics).
        """
        if isinstance(mesh_gen[0], list):
            mesh_gen = mesh_gen[0]

        gen_pts = mesh2pclouds(mesh_gen, num_points=self.num_points)  # [B, P, 3]
        ref_pts = mesh2pclouds(mesh_ref, num_points=self.num_points)  # [B, P, 3]

        for i in range(gen_pts.shape[0]):
            self.gen_pcs.append(gen_pts[i].cpu())
            self.ref_pcs.append(ref_pts[i].cpu())

        return {}

    def compute(self, key: str) -> Dict[str, float]:
        """Compute COV, MMD, 1-NNA with CD distances.

        Returns:
            Dict with 3 entries: {key}_{cov,mmd,1nna}_cd.
        """
        if not self.gen_pcs or not self.ref_pcs:
            return {}

        N_s = len(self.gen_pcs)  # generated (sample)
        N_r = len(self.ref_pcs)  # reference

        logger.info(
            f"GenerationMetrics.compute: N_gen={N_s}, N_ref={N_r}, "
            f"computing pairwise distance matrices..."
        )

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        M_rs_cd = _pairwise_cd_matrix(self.ref_pcs, self.gen_pcs, device=device)
        M_rr_cd = _pairwise_cd_matrix(self.ref_pcs, self.ref_pcs, device=device)
        M_ss_cd = _pairwise_cd_matrix(self.gen_pcs, self.gen_pcs, device=device)

        cov_cd, mmd_cd = _cov_mmd(M_rs_cd.t())  # _cov_mmd expects [N_gen, N_ref]
        nna_cd = _one_nna(M_rr_cd, M_rs_cd, M_ss_cd)

        return {
            f"{key}_cov_cd": cov_cd,
            f"{key}_mmd_cd": mmd_cd,
            f"{key}_1nna_cd": nna_cd,
        }


metrics_dict = {
    "iou": IoU,
    "fid": partial(FID, feature=2048, normalize=True),
    "l1": partial(VolumeMetric, loss_type="l1"),
    "l2": partial(VolumeMetric, loss_type="l2"),
    "chamfer": ChamferDistance,
    "tmd": TMD,
    "gen_metrics": GenerationMetrics,
}


class Metrics(nn.ModuleDict):
    def __init__(self, metrics_list: List[str]):
        super().__init__()
        self.curr_metrics = {}
        self.metrics_list = metrics_list

        for metric_str in metrics_list:
            assert metric_str in metrics_dict, f"Metric {metric_str} not implemented"
            setattr(self, metric_str, metrics_dict[metric_str]())

    @torch.no_grad()
    def update(
        self,
        field_pred,
        volume_field_tgt,
        mesh_field_tgt=None,
        num_views_for_fid: Optional[int] = None,
    ):
        from seen2scene.models.fields import TSDFPatch

        if not isinstance(field_pred, list):
            field_pred = [field_pred]
        device = field_pred[0].device

        if hasattr(self, "l1") or hasattr(self, "l2"):
            assert isinstance(
                volume_field_tgt, TSDFPatch
            ), "Volume metric only supports TSDFPatch"
            volumes_gen = [x.to_dense() for x in field_pred]
            volumes_gt = volume_field_tgt.to_dense()
            band_mask = volume_field_tgt.band_mask.float()

            l1_metrics = self.l1.update(
                volume_gen=volumes_gen, volume_gt=volumes_gt, mask=band_mask
            )
            l2_metrics = self.l2.update(
                volume_gen=volumes_gen, volume_gt=volumes_gt, mask=band_mask
            )
            self.curr_metrics.update(l1_metrics)
            self.curr_metrics.update(l2_metrics)

        # mesh-based metrics (iou, chamfer, tmd)
        needs_gen_meshes = (
            hasattr(self, "iou")
            or hasattr(self, "chamfer")
            or hasattr(self, "tmd")
            or hasattr(self, "gen_metrics")
        )
        needs_tgt_meshes = (
            hasattr(self, "iou")
            or hasattr(self, "chamfer")
            or hasattr(self, "gen_metrics")
        )

        if needs_gen_meshes:
            meshes_gen = [x.export_mesh() for x in field_pred]
        if needs_tgt_meshes:
            assert mesh_field_tgt is not None
            meshes_tgt = mesh_field_tgt.export_mesh()

        if hasattr(self, "iou"):
            iou_metrics = self.iou.update(
                mesh_gen=meshes_gen,
                mesh_tgt=meshes_tgt,
                bbox_world=mesh_field_tgt.bbox_world.cpu().numpy(),
                voxel_size=mesh_field_tgt.voxel_size,
                device=device,
            )
            self.curr_metrics |= iou_metrics

        if hasattr(self, "chamfer"):
            chamfer_metrics = self.chamfer.update(
                mesh_gen=meshes_gen,
                mesh_tgt=meshes_tgt,
                device=device,
            )
            self.curr_metrics |= chamfer_metrics

        if hasattr(self, "tmd"):
            tmd_metrics = self.tmd.update(
                mesh_gen=meshes_gen,
                device=device,
            )
            self.curr_metrics |= tmd_metrics

        if hasattr(self, "gen_metrics"):
            self.gen_metrics.update(
                mesh_gen=meshes_gen,
                mesh_ref=meshes_tgt,
                device=device,
            )

        # image metrics
        if hasattr(self, "fid"):
            assert mesh_field_tgt is not None
            # Number of variations influences FID, so we only use the first variation
            images_gen = field_pred[0].render_field(num_views=num_views_for_fid)
            images_gt = mesh_field_tgt.render_field(num_views=num_views_for_fid)

            images_gen = torch.from_numpy(images_gen).to(device)
            images_gt = torch.from_numpy(images_gt).to(device)

            images_gen = images_gen.flatten(0, 1)
            images_gt = images_gt.flatten(0, 1)

            self.fid.update(images_gen, real=False)
            self.fid.update(images_gt, real=True)

    def export_to_json(self, scene_names: List[str], out_dir: str):
        for idx, scene_name in enumerate(scene_names):
            out_dir_i = os.path.join(out_dir, scene_name)
            os.makedirs(out_dir_i, exist_ok=True)

            metrics_i = {k: v[idx] for k, v in self.curr_metrics.items()}

            summary_path = os.path.join(out_dir_i, "metrics.json")
            with open(summary_path, "w") as f:
                json.dump(metrics_i, f, indent=2)

    def reset(self):
        for metric in self.values():
            metric.reset()

    def compute(self, prefix: str = ""):
        outs = {}
        for metric_str in self.metrics_list:
            metric = getattr(self, metric_str)
            metric_results = {
                prefix + k: v for k, v in metric.compute(metric_str).items()
            }
            outs.update(metric_results)

        return outs
