"""Compute DINO-FID between generated meshes and GT meshes.

Same three-stage workflow as compute_fid.py but uses DINOv2 (ViT-B/14)
features instead of InceptionV3 for the Fréchet distance computation.

Stages 1a/1b/2 are identical — only Stage 3 (compute) is reimplemented.

Usage:
    python -m seen2scene.eval.compute_fid_dino export-gt-mesh-for-sg-baseline \
        --csv-path assets/val_patches_all_3D-FRONT_4000_part2.csv \
        --mesh-dir /tmp/gt_meshes

    python -m seen2scene.eval.compute_fid_dino export-gt-imgs \
        --mesh-dir /tmp/gt_meshes \
        --img-dir /tmp/gt_imgs

    python -m seen2scene.eval.compute_fid_dino export-pred \
        --export-dir /path/to/exports \
        --img-dir /tmp/pred_imgs

    python -m seen2scene.eval.compute_fid_dino compute \
        --pred-img-dir /tmp/pred_imgs \
        --gt-img-dir /tmp/gt_imgs \
        --output results/fid_dino.json
"""

import dataclasses
import glob
import json
import os
from typing import Optional

import numpy as np
import torch
import torchvision.transforms as T
import tyro
from PIL import Image
from scipy.linalg import sqrtm
from tqdm import tqdm

from seen2scene.configs.opt import LOG_ROOT
from seen2scene.eval.compute_fid import (
    ExportGtImages,
    ExportGtMesh_For_SG_Ablation_And_SC,
    ExportGtMesh_For_SG_Baseline,
    ExportPred,
    RenderCfg,
    _export_pred_blockfusion,
    _export_pred_from_ply,
    _export_pred_ours,
    _export_single_gt_images,
    _export_single_gt_mesh,
    _iter_image_batches,
    _save_mesh_images,
    load_csv_rows,
    scene_key,
)
from seen2scene.tools.slurm_utils import Slurm, submit_jobs

# ---------------------------------------------------------------------------
# DINOv2 model loading
# ---------------------------------------------------------------------------

# Ensure torch.hub downloads go to LOG_ROOT/checkpoints/hub/ instead of ~/.cache
os.environ["TORCH_HOME"] = os.path.join(LOG_ROOT, "checkpoints")


def _load_dinov2(device):
    """Load DINOv2 ViT-B/14 model from torch.hub.

    Downloads the checkpoint to LOG_ROOT/checkpoints/hub/ (via TORCH_HOME).
    Returns the model in eval mode. Feature dim: 768 (CLS token).
    """
    print("Loading DINOv2 ViT-B/14 ...")
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
    model = model.to(device).eval()
    print("  DINOv2 loaded (ViT-B/14, 768-dim CLS token features).")
    return model


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

# ImageNet normalization used by DINOv2
_DINO_TRANSFORM = T.Compose(
    [
        T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def _extract_dino_features(model, img_dir, batch_size, device, max_samples=-1):
    """Extract DINOv2 CLS token features from all PNGs under img_dir.

    Args:
        max_samples: If > 0, randomly subsample this many images before extraction.
            -1 uses all images.

    Returns np.ndarray of shape [N, 768].
    """
    paths = sorted(glob.glob(os.path.join(img_dir, "**", "*.png"), recursive=True))
    n = len(paths)
    if n == 0:
        raise ValueError(f"No images found in {img_dir}")
    if max_samples > 0 and max_samples < n:
        rng = np.random.default_rng(seed=42)
        paths = list(rng.choice(paths, size=max_samples, replace=False))
        print(f"  {n} images in {img_dir}, subsampled to {len(paths)}")
    else:
        print(f"  {n} images from {img_dir}")

    all_feats = []
    batch = []
    for p in tqdm(paths, desc="  DINO features"):
        try:
            img = Image.open(p).convert("RGB")
        except Exception as e:
            print(f"WARNING: skipping corrupted image {p}: {e}")
            continue
        batch.append(_DINO_TRANSFORM(img))
        if len(batch) == batch_size:
            imgs_t = torch.stack(batch).to(device)
            with torch.inference_mode():
                feats = model(imgs_t)  # [B, 768]
            all_feats.append(feats.cpu().numpy())
            batch = []
    if batch:
        imgs_t = torch.stack(batch).to(device)
        with torch.inference_mode():
            feats = model(imgs_t)
        all_feats.append(feats.cpu().numpy())

    return np.concatenate(all_feats, axis=0)


# ---------------------------------------------------------------------------
# Fréchet distance (same formula as compute_fpd.py)
# ---------------------------------------------------------------------------


def frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Fréchet distance between two multivariate Gaussians."""
    diff = mu1 - mu2
    covmean, _ = sqrtm(sigma1 @ sigma2, disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = sqrtm((sigma1 + offset) @ (sigma2 + offset))
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            print(f"Warning: imaginary component {m:.6f}")
        covmean = covmean.real
    return float(
        diff @ diff + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean)
    )


# ---------------------------------------------------------------------------
# DINO-FID computation
# ---------------------------------------------------------------------------


def compute_fid_dino(
    pred_img_dir, gt_img_dir, batch_size=64, device=None, max_samples=-1
):
    """Compute FID using DINOv2 features between two image directories.

    Args:
        max_samples: If > 0, randomly subsample this many images per directory.
            -1 uses all images.
    """
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    model = _load_dinov2(device)

    print("Extracting pred features ...")
    pred_feats = _extract_dino_features(
        model, pred_img_dir, batch_size, device, max_samples
    )
    print("Extracting GT features ...")
    gt_feats = _extract_dino_features(
        model, gt_img_dir, batch_size, device, max_samples
    )

    mu_pred = np.mean(pred_feats, axis=0)
    sigma_pred = np.cov(pred_feats, rowvar=False)
    mu_gt = np.mean(gt_feats, axis=0)
    sigma_gt = np.cov(gt_feats, rowvar=False)

    print("Computing Fréchet distance ...")
    return frechet_distance(mu_pred, sigma_pred, mu_gt, sigma_gt)


# ---------------------------------------------------------------------------
# Config dataclass (only Compute is new; export stages reuse compute_fid.py)
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class Compute:
    """Compute DINO-FID from two image directories."""

    pred_img_dir: str
    gt_img_dir: str
    batch_size: int = 512
    max_samples: int = -1
    """Max images to use per directory. -1 uses all images."""
    output: Optional[str] = None
    slurm: Slurm = dataclasses.field(
        default_factory=lambda: Slurm(
            slurm_job_name="fid_dino_compute", gpus_per_node=1, nodes=1
        )
    )


def _run_compute(cfg: Compute):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    fid_value = compute_fid_dino(
        cfg.pred_img_dir, cfg.gt_img_dir, cfg.batch_size, device, cfg.max_samples
    )
    print(f"\nDINO-FID: {fid_value:.4f}")
    if cfg.output:
        os.makedirs(os.path.dirname(cfg.output) or ".", exist_ok=True)
        with open(cfg.output, "w") as f:
            json.dump(
                {
                    "fid_dino": fid_value,
                    "encoder": "dinov2_vitb14",
                    "pred_img_dir": cfg.pred_img_dir,
                    "gt_img_dir": cfg.gt_img_dir,
                },
                f,
                indent=2,
            )
        print(f"Saved to {cfg.output}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    cfg = tyro.cli(
        ExportGtMesh_For_SG_Baseline
        | ExportGtMesh_For_SG_Ablation_And_SC
        | ExportGtImages
        | ExportPred
        | Compute
    )

    if isinstance(cfg, ExportGtMesh_For_SG_Baseline):
        rows = load_csv_rows(cfg.csv_path)
        submit_jobs(
            fn=_export_single_gt_mesh,
            fn_kwargs_list=[
                {
                    "dataset": r["dataset"],
                    "scene_name": r["scene_name"],
                    "bbox": r["bbox"],
                }
                for r in rows
            ],
            fn_kwargs_share={"mesh_dir": cfg.mesh_dir},
            slurm_kwargs=dataclasses.asdict(cfg.slurm),
        )

    elif isinstance(cfg, ExportGtMesh_For_SG_Ablation_And_SC):
        rows = load_csv_rows(cfg.csv_path)
        submit_jobs(
            fn=_export_single_gt_mesh,
            fn_kwargs_list=[
                {
                    "dataset": r["dataset"],
                    "scene_name": r["scene_name"],
                    "bbox": r["bbox"],
                }
                for r in rows
            ],
            fn_kwargs_share={"mesh_dir": cfg.mesh_dir},
            slurm_kwargs=dataclasses.asdict(cfg.slurm),
        )

    elif isinstance(cfg, ExportGtImages):
        keys = sorted(
            d
            for d in os.listdir(cfg.mesh_dir)
            if os.path.exists(os.path.join(cfg.mesh_dir, d, "mesh.ply"))
        )
        submit_jobs(
            fn=_export_single_gt_images,
            fn_kwargs_list=[{"key": k} for k in keys],
            fn_kwargs_share={
                "mesh_dir": cfg.mesh_dir,
                "img_dir": cfg.img_dir,
                "num_views": cfg.render.num_views,
                "thetas": cfg.render.thetas,
                "resolution": cfg.render.resolution,
                "ceiling_clip": cfg.render.ceiling_clip,
            },
            slurm_kwargs=dataclasses.asdict(cfg.slurm),
        )

    elif isinstance(cfg, ExportPred):
        render_share = {
            "img_dir": cfg.img_dir,
            "num_views": cfg.render.num_views,
            "thetas": cfg.render.thetas,
            "resolution": cfg.render.resolution,
            "ceiling_clip": cfg.render.ceiling_clip,
        }
        if cfg.method == "ours":
            pattern = os.path.join(cfg.export_dir, "*", "mesh_tsdf (Generation)_0.ply")
            ply_paths = sorted(glob.glob(pattern))
            print(f"Found {len(ply_paths)} PLY files in {cfg.export_dir}")
            submit_jobs(
                fn=_export_pred_ours,
                fn_kwargs_list=[{"ply_path": p} for p in ply_paths],
                fn_kwargs_share=render_share,
                slurm_kwargs=dataclasses.asdict(cfg.slurm),
            )
        elif cfg.method == "blockfusion":
            ply_paths = sorted(glob.glob(os.path.join(cfg.export_dir, "*.ply")))
            print(f"Found {len(ply_paths)} PLY files in {cfg.export_dir}")
            submit_jobs(
                fn=_export_pred_blockfusion,
                fn_kwargs_list=[{"ply_path": p} for p in ply_paths],
                fn_kwargs_share=render_share,
                slurm_kwargs=dataclasses.asdict(cfg.slurm),
            )
        else:  # lt3sd, worldgrow
            ply_paths = sorted(glob.glob(os.path.join(cfg.export_dir, "*.ply")))
            print(f"Found {len(ply_paths)} PLY files in {cfg.export_dir}")
            submit_jobs(
                fn=_export_pred_from_ply,
                fn_kwargs_list=[{"ply_path": p} for p in ply_paths],
                fn_kwargs_share={**render_share, "method": cfg.method},
                slurm_kwargs=dataclasses.asdict(cfg.slurm),
            )

    elif isinstance(cfg, Compute):
        submit_jobs(
            fn=_run_compute,
            fn_kwargs_share={"cfg": cfg},
            slurm_kwargs=dataclasses.asdict(cfg.slurm),
        )


if __name__ == "__main__":
    main()
