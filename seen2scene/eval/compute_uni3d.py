"""Compute Uni3D Fréchet Point Cloud Distance (FPD) between generated and GT meshes.

Meshes are loaded and preprocessed identically to compute_cov_mmd_nna.py
(ceiling clip, renormalization per method), then sampled to 10K-point clouds
and scored via Uni3D embeddings.

Requires the ``uni3d`` conda environment (PyTorch 2.4.1+cu121, pointnet2_ops).

Usage:
    conda activate uni3d

    # Ours
    python -m seen2scene.eval.compute_uni3d \
        --export-dir $LOG/.../patch_generation --gt-mesh-dir /path/to/gt_meshes

    # Blockfusion
    python -m seen2scene.eval.compute_uni3d \
        --export-dir /path/to/blockfusion_exports --gt-mesh-dir /path/to/gt_meshes \
        --method blockfusion

    # LT3SD
    python -m seen2scene.eval.compute_uni3d \
        --export-dir /path/to/lt3sd_exports --gt-mesh-dir /path/to/gt_meshes \
        --method lt3sd

    # WorldGrow
    python -m seen2scene.eval.compute_uni3d \
        --export-dir /path/to/worldgrow_exports --gt-mesh-dir /path/to/gt_meshes \
        --method worldgrow

    # SGNN
    python -m seen2scene.eval.compute_uni3d \
        --export-dir /path/to/sgnn_exports --gt-mesh-dir /path/to/gt_meshes \
        --method sgnn

    # NKSR
    python -m seen2scene.eval.compute_uni3d \
        --export-dir /path/to/nksr_exports --gt-mesh-dir /path/to/gt_meshes \
        --method nksr
"""

import sys
import os

# Ensure the project root is on sys.path so `seen2scene` is importable
# even when running as `python seen2scene/eval/compute_uni3d.py`.
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Uni3D lives outside this repo; set UNI3D_ROOT so we can import uni3d_eval.
_UNI3D_ROOT = os.environ.get("UNI3D_ROOT")
if _UNI3D_ROOT and _UNI3D_ROOT not in sys.path:
    sys.path.insert(0, _UNI3D_ROOT)

import dataclasses
import glob
import json
import time
from typing import Literal, Optional

import numpy as np
import trimesh
import tyro
from tqdm import tqdm

from seen2scene.tools.common_utils import clip_mesh
from seen2scene.tools.renorm import (
    apply_ceiling_clip,
    parse_bbox_from_stem,
    renorm_blockfusion,
    renorm_lt3sd_sg,
    renorm_worldgrow,
)
from uni3d_eval import compute_uni3d_fd, extract_features

UNI3D_NUM_POINTS = 10_000  # Uni3D expects 10K-point clouds


@dataclasses.dataclass
class Config:
    """Compute Uni3D FPD between generated and GT meshes."""

    export_dir: str
    """For ours: patch_generation directory. For baselines: PLY export directory."""

    gt_mesh_dir: str
    """Directory containing GT mesh PLY files."""

    method: Literal["ours", "blockfusion", "lt3sd", "worldgrow", "sgnn", "nksr"] = "ours"
    """Method whose predictions to evaluate."""

    num_points: int = UNI3D_NUM_POINTS
    """Number of points to sample from each mesh (default 10000 for Uni3D)."""

    device: str = "cuda:0"
    batch_size: int = 32
    """Batch size for Uni3D feature extraction."""

    max_samples: Optional[int] = None
    """Cap the number of samples used from each set (gen and GT). None = use all."""

    output: Optional[str] = None
    """Path to save results as JSON."""

    def __post_init__(self):
        if self.output is None:
            self.output = f"{self.method}_uni3d_fpd.json"


# ---------------------------------------------------------------------------
# Mesh -> numpy point cloud sampling
# ---------------------------------------------------------------------------


def _sample_pc(mesh, num_points):
    """Sample a point cloud from a mesh. Returns np.ndarray of shape [num_points, 3]."""
    points, _ = trimesh.sample.sample_surface(mesh, num_points)
    return points.astype(np.float32)


# ---------------------------------------------------------------------------
# GT loader (same logic as compute_cov_mmd_nna)
# ---------------------------------------------------------------------------


def _load_gt_pcs(gt_mesh_dir, num_points, max_samples=None):
    """Load PLY files from gt_mesh_dir and convert to point clouds."""
    pcs = []
    ply_paths = sorted(glob.glob(os.path.join(gt_mesh_dir, "*", "mesh.ply")))
    for ply in tqdm(ply_paths, desc="Loading GT"):
        mesh = trimesh.load(ply, process=False)
        if len(mesh.vertices) == 0:
            continue
        scene_dir = os.path.dirname(ply)
        meta_files = sorted(glob.glob(os.path.join(scene_dir, "meta_*.json")))
        if meta_files:
            with open(meta_files[0]) as f:
                meta = json.load(f)
            bbox = apply_ceiling_clip(meta["bbox_world"])
        else:
            b = mesh.bounds
            bbox = apply_ceiling_clip(
                [b[0][0], b[0][1], b[0][2], b[1][0], b[1][1], b[1][2]]
            )
        mesh = clip_mesh(mesh, bbox)
        if mesh is not None and len(mesh.vertices) > 0:
            pcs.append(_sample_pc(mesh, num_points))
            if max_samples and len(pcs) >= max_samples:
                break
    return pcs


# ---------------------------------------------------------------------------
# Generated mesh loaders (same logic as compute_cov_mmd_nna)
# ---------------------------------------------------------------------------


def _load_gen_ours(export_dir, num_points, max_samples=None):
    pcs = []
    pattern = os.path.join(export_dir, "*", "mesh_tsdf (Generation)_0.ply")
    for ply in tqdm(sorted(glob.glob(pattern)), desc="Loading gen (ours)"):
        mesh = trimesh.load(ply, process=False)
        if len(mesh.vertices) == 0:
            continue
        scene_dir = os.path.dirname(ply)
        meta_files = sorted(glob.glob(os.path.join(scene_dir, "meta_*.json")))
        if meta_files:
            with open(meta_files[0]) as f:
                meta = json.load(f)
            bbox = apply_ceiling_clip(meta["bbox_world"])
            mesh = clip_mesh(mesh, bbox)
            if mesh is None or len(mesh.vertices) == 0:
                continue
        pcs.append(_sample_pc(mesh, num_points))
        if max_samples and len(pcs) >= max_samples:
            break
    return pcs


def _load_gen_blockfusion(export_dir, num_points, max_samples=None):
    pcs = []
    for ply in tqdm(
        sorted(glob.glob(os.path.join(export_dir, "*.ply"))),
        desc="Loading gen (blockfusion)",
    ):
        mesh = trimesh.load(ply, process=False)
        if len(mesh.vertices) == 0:
            continue
        parsed = parse_bbox_from_stem(os.path.splitext(os.path.basename(ply))[0])
        if parsed is None:
            continue
        _, csv_bbox = parsed
        mesh, render_bbox = renorm_blockfusion(mesh, csv_bbox)
        mesh = clip_mesh(mesh, render_bbox)
        if mesh is not None and len(mesh.vertices) > 0:
            pcs.append(_sample_pc(mesh, num_points))
            if max_samples and len(pcs) >= max_samples:
                break
    return pcs


def _load_gen_lt3sd(export_dir, num_points, max_samples=None):
    pcs = []
    for ply in tqdm(
        sorted(glob.glob(os.path.join(export_dir, "*.ply"))), desc="Loading gen (lt3sd)"
    ):
        mesh = trimesh.load(ply, process=False)
        if len(mesh.vertices) == 0:
            continue
        for sub_mesh, sub_bbox in renorm_lt3sd_sg(mesh):
            pcs.append(_sample_pc(sub_mesh, num_points))
            if max_samples and len(pcs) >= max_samples:
                break
        if max_samples and len(pcs) >= max_samples:
            break
    return pcs


def _load_gen_worldgrow(export_dir, num_points, max_samples=None):
    pcs = []
    for ply in tqdm(
        sorted(glob.glob(os.path.join(export_dir, "*.ply"))),
        desc="Loading gen (worldgrow)",
    ):
        mesh = trimesh.load(ply, process=False)
        if len(mesh.vertices) == 0:
            continue
        result = renorm_worldgrow(mesh)
        if result is None:
            continue
        mesh, crop_bbox = result
        pcs.append(_sample_pc(mesh, num_points))
        if max_samples and len(pcs) >= max_samples:
            break
    return pcs


SGNN_VOXEL_SIZE = 0.011  # same as compute_iou_l1_l2_tmd.DEFAULT_VOXEL_SIZE


def _load_gen_sgnn(export_dir, num_points, max_samples=None):
    """SGNN: flat files ``{scene}_{bbox6}pred.ply``, mesh in voxel coords."""
    pcs = []
    for ply in tqdm(
        sorted(glob.glob(os.path.join(export_dir, "*pred.ply"))),
        desc="Loading gen (sgnn)",
    ):
        mesh = trimesh.load(ply, process=False, force='mesh')
        if len(mesh.vertices) == 0:
            continue
        stem = os.path.splitext(os.path.basename(ply))[0]  # strip .ply
        stem = stem.replace("pred", "")  # strip trailing "pred"
        parsed = parse_bbox_from_stem(stem)
        if parsed is None:
            continue
        _, csv_bbox = parsed
        # SGNN outputs voxel coords with swapped X/Z axes
        mesh.vertices[:, [0, 2]] = mesh.vertices[:, [2, 0]]
        # Voxel coords → world coords
        mesh.vertices = mesh.vertices * SGNN_VOXEL_SIZE + np.array(csv_bbox[:3])
        clip_bbox = apply_ceiling_clip(csv_bbox)
        mesh = clip_mesh(mesh, clip_bbox)
        if mesh is not None and len(mesh.vertices) > 0:
            pcs.append(_sample_pc(mesh, num_points))
            if max_samples and len(pcs) >= max_samples:
                break
    return pcs


def _load_gen_nksr(export_dir, num_points, max_samples=None):
    """NKSR: flat files ``{scene}_{bbox6}pred.ply``, mesh already in world coords."""
    pcs = []
    for ply in tqdm(
        sorted(glob.glob(os.path.join(export_dir, "*pred.ply"))),
        desc="Loading gen (nksr)",
    ):
        mesh = trimesh.load(ply, process=False, force='mesh')
        if len(mesh.vertices) == 0:
            continue
        stem = os.path.splitext(os.path.basename(ply))[0]
        stem = stem.replace("pred", "")
        parsed = parse_bbox_from_stem(stem)
        if parsed is None:
            continue
        _, csv_bbox = parsed
        clip_bbox = apply_ceiling_clip(csv_bbox)
        mesh = clip_mesh(mesh, clip_bbox)
        if mesh is not None and len(mesh.vertices) > 0:
            pcs.append(_sample_pc(mesh, num_points))
            if max_samples and len(pcs) >= max_samples:
                break
    return pcs


_GEN_LOADERS = {
    "ours": _load_gen_ours,
    "blockfusion": _load_gen_blockfusion,
    "lt3sd": _load_gen_lt3sd,
    "worldgrow": _load_gen_worldgrow,
    "sgnn": _load_gen_sgnn,
    "nksr": _load_gen_nksr,
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    cfg = tyro.cli(Config)

    print(f"Method: {cfg.method}, export_dir: {cfg.export_dir}")
    print(f"Sampling {cfg.num_points} points per mesh for Uni3D")

    # Load meshes -> point clouds (numpy)
    t0 = time.time()
    gen_pcs = _GEN_LOADERS[cfg.method](cfg.export_dir, cfg.num_points, cfg.max_samples)
    print(f"Loaded {len(gen_pcs)} generated point clouds in {time.time() - t0:.1f}s")

    t0 = time.time()
    gt_pcs = _load_gt_pcs(cfg.gt_mesh_dir, cfg.num_points, cfg.max_samples)
    print(f"Loaded {len(gt_pcs)} GT point clouds in {time.time() - t0:.1f}s")

    if len(gen_pcs) < 2 or len(gt_pcs) < 2:
        print("Need at least 2 samples in each set. Aborting.")
        return

    # Extract features
    print(f"Extracting Uni3D features (batch_size={cfg.batch_size})...")
    t0 = time.time()
    gen_feats = extract_features(gen_pcs, device=cfg.device, batch_size=cfg.batch_size)
    gt_feats = extract_features(gt_pcs, device=cfg.device, batch_size=cfg.batch_size)
    print(f"  Features extracted in {time.time() - t0:.1f}s")
    print(f"  Gen features: {gen_feats.shape}, GT features: {gt_feats.shape}")

    # Compute FPD
    fpd = compute_uni3d_fd(gen_feats, gt_feats)

    print(f"\nResults:")
    print(f"  Uni3D FPD: {fpd:.4f}")

    if cfg.output:
        os.makedirs(os.path.dirname(cfg.output) or ".", exist_ok=True)
        with open(cfg.output, "w") as f:
            json.dump(
                {
                    "method": cfg.method,
                    "n_gen": len(gen_pcs),
                    "n_ref": len(gt_pcs),
                    "num_points": cfg.num_points,
                    "uni3d_fpd": fpd,
                },
                f,
                indent=2,
            )
        print(f"Saved to {cfg.output}")


if __name__ == "__main__":
    main()
