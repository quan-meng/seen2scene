"""Compute mesh-based metrics (CD, L1, L2, TMD) from exported PLY/NPZ files.

GT data is loaded from pre-exported gt_dir (mesh.ply, volume.npy, bbox.json).
Pred data is loaded from gen_dir scene subdirectories.

Usage:
    python -m seen2scene.eval.compute_iou_l1_l2_tmd \
        --gen_dir /path/to/scene_outputs \
        --gt_dir /path/to/gt_for_sg_ablation_and_sc \
        --metrics cd l1 l2 tmd \
        --device cuda:0 --num_workers 4
"""

import dataclasses
import faulthandler
import glob
import json
import logging
import os
import re
from typing import Any, Dict, List, Literal, Optional

import numpy as np
import torch
import torch.utils.data

# IMPORTANT: import metrics (which pulls in torchmetrics) BEFORE trimesh
# to avoid a segfault caused by C-extension load-order conflict.
from seen2scene.tools.metrics import ChamferDistance, TMD, _validate_mesh

import trimesh
import tyro
from tqdm import tqdm

logger = logging.getLogger(__name__)


_DATASET_PATTERNS = {
    "3D-FRONT": re.compile(
        r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}($|_)"
    ),
    "ScannetPP": re.compile(r"^[0-9a-f]{10}($|_)"),
    "ARKitScenes": re.compile(r"^\d+($|_)"),
}

DEFAULT_VOXEL_SIZE = 0.011
MIN_PRED_VERTICES = (
    10000  # Skip degenerate predictions (SGNN occasionally produces near-empty meshes)
)


def _bbox_min_key(bbox):
    """Round bbox_min (first 3 values) to 1 decimal for matching against directory/file names."""
    return "_".join(f"{v:.1f}" for v in bbox[:3])


GT_DIR_DEFAULT = os.environ.get("SEEN2SCENE_GT_DIR", "gt_for_sg_ablation_and_sc")


@dataclasses.dataclass
class Config:
    """Compute mesh-based metrics (IoU, L1, L2, TMD) from exported PLY/NPZ files."""

    gen_dir: str
    """Root directory containing scene subdirectories with predictions."""
    gt_dir: str = GT_DIR_DEFAULT
    """Directory with pre-exported GT (mesh.ply, volume.npy, bbox.json per scene)."""
    method: Literal["ours", "sgnn", "nksr"] = "ours"
    """Method whose predictions to evaluate."""
    metrics: List[Literal["cd", "l1", "l2", "tmd"]] = dataclasses.field(
        default_factory=lambda: ["cd", "l1", "l2", "tmd"]
    )
    device: str = "cuda:0"
    dataset: Optional[Literal["3D-FRONT", "ScannetPP", "ARKitScenes"]] = None
    """Restrict to scenes matching this dataset's name pattern."""
    max_samples: Optional[int] = None
    """Cap the number of scenes to evaluate. None = use all."""
    export_pair: Optional[str] = None
    """Export one GT+pred mesh pair to this directory for debugging, then exit."""
    output: Optional[str] = None
    """Path to save aggregated results as JSON."""
    resume: bool = False
    """Skip scenes already present in the output file (requires --output)."""
    num_workers: int = 4
    """Number of DataLoader workers for parallel I/O."""


# ---------------------------------------------------------------------------
# Meta helpers
# ---------------------------------------------------------------------------


def load_meta(scene_dir):
    meta_files = glob.glob(os.path.join(scene_dir, "*meta*.json"))
    if not meta_files:
        return None
    with open(meta_files[0]) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# GT loaders (from pre-exported gt_dir)
# ---------------------------------------------------------------------------


def load_gt_mesh(gt_scene_dir):
    """Load GT mesh.ply from pre-exported gt_dir scene subdirectory."""
    mesh_path = os.path.join(gt_scene_dir, "mesh.ply")
    if not os.path.exists(mesh_path):
        return None
    mesh = trimesh.load(mesh_path, process=False)
    if not _validate_mesh(mesh, label=f"GT:{os.path.basename(gt_scene_dir)}"):
        return None
    return mesh


def load_gt_volume(gt_scene_dir, truncation):
    """Load GT volume.npy from pre-exported gt_dir scene subdirectory."""
    vol_path = os.path.join(gt_scene_dir, "volume.npy")
    if not os.path.exists(vol_path):
        return None
    vol = np.load(vol_path).astype(np.float32)
    vol = np.clip(vol, -truncation, truncation)
    return vol


# ---------------------------------------------------------------------------
# Pred loaders (from gen_dir)
# ---------------------------------------------------------------------------


def npz_to_dense(npz_path, patch_shape, truncation):
    """Reconstruct a dense TSDF volume from a sparse NPZ export."""
    data = np.load(npz_path)
    volume = np.full(patch_shape, truncation, dtype=np.float32)
    if "unknown" in data and len(data["unknown"]) > 0:
        unk = data["unknown"]
        volume[unk[:, 0], unk[:, 1], unk[:, 2]] = -truncation
    ijks, values = data["ijks"], data["values"]
    if values.ndim > 1:
        values = values[:, 0]
    if len(ijks) > 0:
        volume[ijks[:, 0], ijks[:, 1], ijks[:, 2]] = values
    return volume


def load_pred_meshes(scene_dir, method, bbox=None, voxel_size=DEFAULT_VOXEL_SIZE):
    """Load predicted mesh(es). Returns list[trimesh] or None."""
    if method == "ours":
        gen_paths = sorted(
            glob.glob(os.path.join(scene_dir, "mesh_tsdf (Generation)_*.ply")),
            key=lambda p: int(re.search(r"_(\d+)\.ply$", p).group(1)),
        )
        meshes = [trimesh.load(p, process=False) for p in gen_paths]
        meshes = [
            m
            for m in meshes
            if _validate_mesh(m, label=f"pred:{os.path.basename(scene_dir)}")
        ]
        return meshes or None
    elif method == "sgnn":
        # For flat-file baselines, scene_dir is the PLY path itself
        if not os.path.isfile(scene_dir):
            return None
        mesh = trimesh.load(scene_dir, process=False, force="mesh")
        if not _validate_mesh(mesh, label=f"pred:{os.path.basename(scene_dir)}"):
            return None
        if len(mesh.vertices) < MIN_PRED_VERTICES:
            logger.warning(
                f"SGNN mesh too sparse ({len(mesh.vertices)} verts), skipping: {os.path.basename(scene_dir)}"
            )
            return None
        # SGNN outputs voxel coords with swapped X/Z axes
        verts = mesh.vertices.copy()
        verts[:, [0, 2]] = verts[:, [2, 0]]
        # Renormalize: voxel coords → world coords
        if bbox is not None:
            verts = verts * voxel_size + np.array(bbox[:3])
        mesh.vertices = verts
        return [mesh]
    elif method == "nksr":
        # Flat-file layout like SGNN, but already in world coords (no transform)
        if not os.path.isfile(scene_dir):
            return None
        mesh = trimesh.load(scene_dir, process=False, force="mesh")
        if not _validate_mesh(mesh, label=f"pred:{os.path.basename(scene_dir)}"):
            return None
        if len(mesh.vertices) == 0:
            return None
        return [mesh]
    else:
        raise ValueError(f"Unknown method: {method}")


def load_pred_volume(scene_dir, method, patch_shape, truncation, voxel_size):
    """Load predicted TSDF as a dense float32 array, or None."""
    if method == "ours":
        # Dense .npy volumes (e.g. "volume_tsdf (Generation)_0.npy")
        npy_files = sorted(
            glob.glob(os.path.join(scene_dir, "volume_tsdf (Generation)_*.npy"))
        )
        if npy_files:
            vol = np.load(npy_files[0]).astype(np.float32).squeeze()
            vol = np.clip(vol, -truncation, truncation)
            return vol
    elif method == "sgnn":
        # Dense .npy volume in voxel-space TSDF (values in [-3, 3])
        npy_path = scene_dir.replace("pred.ply", "pred.npy")
        if not os.path.exists(npy_path):
            return None
        vol = np.load(npy_path).astype(np.float32)
        # SGNN has swapped X/Z axes
        vol = np.swapaxes(vol, 0, 2)
        # Convert voxel-space TSDF → world-space TSDF.
        # SGNN uses inverted sign convention (negative = free space) vs. GT
        # (positive = free space), so we negate before scaling.
        vol = -vol * voxel_size
        vol = np.clip(vol, -truncation, truncation)
        # Handle minor shape mismatch (e.g., 256 vs 255 from rounding)
        if vol.shape != patch_shape:
            out = np.full(patch_shape, truncation, dtype=np.float32)
            slices = tuple(
                slice(0, min(s, ps)) for s, ps in zip(vol.shape, patch_shape)
            )
            out[slices] = vol[slices]
            vol = out
        return vol
    elif method == "nksr":
        # NKSR volumes: {stem}pred_tsdf.npy, already in world-space TSDF
        npy_path = scene_dir.replace("pred.ply", "pred_tsdf.npy")
        if not os.path.exists(npy_path):
            return None
        vol = np.load(npy_path).astype(np.float32)
        vol = np.clip(vol, -truncation, truncation)
        return vol
    else:
        raise ValueError(f"Unknown method: {method}")


# ---------------------------------------------------------------------------
# Scene Dataset for DataLoader-based parallel I/O
# ---------------------------------------------------------------------------


class SceneDataset(torch.utils.data.Dataset):
    """Loads GT + pred data for each scene. Heavy I/O happens in workers."""

    def __init__(self, entries: List[Dict[str, Any]], method: str, metrics: List[str]):
        self.entries = entries
        self.method = method
        self.metrics = metrics

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx) -> Dict[str, Any]:
        entry = self.entries[idx]
        scene_dir = entry["scene_dir"]
        gt_scene_dir = entry["gt_scene_dir"]
        bbox = entry["bbox"]
        key = entry["key"]

        meta = load_meta(scene_dir) if self.method == "ours" else None
        voxel_size = meta["voxel_size"] if meta else DEFAULT_VOXEL_SIZE
        truncation = meta.get("truncation", 3 * voxel_size) if meta else 3 * voxel_size

        result = {"key": key, "bbox": bbox, "voxel_size": voxel_size}

        # For flat-file baselines, validate the pred mesh first; skip all metrics
        # if the PLY is empty or degenerate (SGNN/NKSR occasionally produce these).
        if self.method in ("sgnn", "nksr"):
            meshes_gen = load_pred_meshes(
                scene_dir, self.method, bbox=bbox, voxel_size=voxel_size
            )
            if meshes_gen is None:
                return result  # no data → compute_metrics_from_loaded returns None
            result["meshes_gen"] = meshes_gen

        # Load mesh data (for CD / TMD)
        if "cd" in self.metrics or "tmd" in self.metrics:
            mesh_tgt = load_gt_mesh(gt_scene_dir)
            # Reuse mesh already loaded for flat-file baselines; load fresh for "ours"
            if "meshes_gen" not in result:
                meshes_gen = load_pred_meshes(
                    scene_dir, self.method, bbox=bbox, voxel_size=voxel_size
                )
                result["meshes_gen"] = meshes_gen
            else:
                meshes_gen = result["meshes_gen"]
            # Sanity check: verify GT and pred meshes are in the same coordinate system
            if mesh_tgt is not None and meshes_gen:
                gt_min, gt_max = mesh_tgt.vertices.min(axis=0), mesh_tgt.vertices.max(
                    axis=0
                )
                pred_min, pred_max = meshes_gen[0].vertices.min(axis=0), meshes_gen[
                    0
                ].vertices.max(axis=0)
                # Check if bounding boxes overlap at all
                overlap = np.all(gt_min <= pred_max) and np.all(pred_min <= gt_max)
                if not overlap:
                    logger.warning(
                        f"[{key}] Coordinate mismatch! "
                        f"GT range: [{gt_min} .. {gt_max}], "
                        f"Pred range: [{pred_min} .. {pred_max}]"
                    )
            result["mesh_tgt"] = mesh_tgt
            result["meshes_gen"] = meshes_gen  # may already be set; overwrite is fine

        # Load volume data (for L1 / L2)
        if "l1" in self.metrics or "l2" in self.metrics:
            vol_gt = load_gt_volume(gt_scene_dir, truncation)
            if vol_gt is not None:
                vol_pred = load_pred_volume(
                    scene_dir, self.method, vol_gt.shape, truncation, voxel_size
                )
                # Sanity check: verify volume value ranges are comparable
                if vol_pred is not None:
                    gt_range = (float(vol_gt.min()), float(vol_gt.max()))
                    pred_range = (float(vol_pred.min()), float(vol_pred.max()))
                    if abs(gt_range[1] - pred_range[1]) > 10 * truncation:
                        logger.warning(
                            f"[{key}] Volume range mismatch! "
                            f"GT: [{gt_range[0]:.4f}, {gt_range[1]:.4f}], "
                            f"Pred: [{pred_range[0]:.4f}, {pred_range[1]:.4f}]"
                        )
            else:
                vol_pred = None
            result["vol_gt"] = vol_gt
            result["vol_pred"] = vol_pred

        return result


def _passthrough_collate(batch):
    """Return the single item unwrapped (batch_size=1)."""
    return batch[0]


# ---------------------------------------------------------------------------
# Per-scene metric computation (GPU, main thread)
# ---------------------------------------------------------------------------


def compute_metrics_from_loaded(
    data: Dict[str, Any], metrics: List[str], device: str
) -> Optional[Dict[str, float]]:
    """Compute metrics from pre-loaded data. Returns dict or None if data is missing."""
    bbox = data["bbox"]
    voxel_size = data["voxel_size"]
    bbox_world = np.array(bbox).reshape(1, 6)

    scene_metrics = {}

    # --- Mesh-based (CD, TMD) ---
    if "cd" in metrics or "tmd" in metrics:
        mesh_tgt = data.get("mesh_tgt")
        meshes_gen = data.get("meshes_gen")
        if mesh_tgt is not None and meshes_gen:
            # CD only for 3D-FRONT scenes (UUID pattern)
            is_3dfront = _DATASET_PATTERNS["3D-FRONT"].match(data["key"]) is not None
            if "cd" in metrics and is_3dfront:
                result = ChamferDistance().update(
                    mesh_gen=[[m] for m in meshes_gen],
                    mesh_tgt=[mesh_tgt],
                    device=device,
                )
                scene_metrics.update({k: v[0] for k, v in result.items()})

            if "tmd" in metrics and len(meshes_gen) >= 2:
                result = TMD().update(mesh_gen=[[m] for m in meshes_gen], device=device)
                scene_metrics.update(
                    {k: v[0] if isinstance(v, list) else v for k, v in result.items()}
                )

    # --- Volume-based (L1, L2) ---
    if "l1" in metrics or "l2" in metrics:
        vol_gt = data.get("vol_gt")
        vol_pred = data.get("vol_pred")
        if vol_gt is not None and vol_pred is not None:
            vol_gt_t = torch.from_numpy(vol_gt).to(device)
            vol_pred_t = torch.from_numpy(vol_pred).to(device)

            # known mask: voxels not marked as unknown (-truncation) in GT
            truncation = vol_gt_t.min()  # -truncation is the sentinel value
            mask = vol_gt_t > truncation
            if mask.sum() > 0:
                diff = vol_pred_t[mask] - vol_gt_t[mask]
                if "l1" in metrics:
                    scene_metrics["l1"] = float(diff.abs().mean().cpu())
                if "l2" in metrics:
                    scene_metrics["l2"] = float((diff**2).mean().cpu())

    return scene_metrics if scene_metrics else None


# ---------------------------------------------------------------------------
# Partial results for --resume
# ---------------------------------------------------------------------------


def _load_partial_results(output_path):
    """Load per-scene results from partial output file for --resume."""
    partial_path = output_path + ".partial"
    if not os.path.exists(partial_path):
        return {}
    with open(partial_path) as f:
        data = json.load(f)
    return data.get("per_scene", {})


def _save_partial_results(output_path, per_scene):
    """Incrementally save per-scene results to a .partial file."""
    partial_path = output_path + ".partial"
    os.makedirs(os.path.dirname(partial_path) or ".", exist_ok=True)
    with open(partial_path, "w") as f:
        json.dump({"per_scene": per_scene}, f)


# ---------------------------------------------------------------------------
# Scene discovery
# ---------------------------------------------------------------------------


def _load_gt_index(gt_dir):
    """Build index from gt_dir: maps (scene_name, bbox_min_key) → {gt_scene_dir, bbox, dataset}.

    Each GT subdirectory is named ``{scene}_{x0}_{y0}_{z0}_{x1}_{y1}_{z1}`` and
    contains ``bbox.json`` with the precise bbox, dataset, and scene_name.
    """
    index = {}
    for name in sorted(os.listdir(gt_dir)):
        gt_scene_dir = os.path.join(gt_dir, name)
        if not os.path.isdir(gt_scene_dir):
            continue
        bbox_json = os.path.join(gt_scene_dir, "bbox.json")
        if not os.path.exists(bbox_json):
            continue
        with open(bbox_json) as f:
            meta = json.load(f)
        bbox = meta["bbox"]
        scene_name = meta["scene_name"]
        dataset = meta.get("dataset", "")
        key = (scene_name, _bbox_min_key(bbox))
        index[key] = {"gt_scene_dir": gt_scene_dir, "bbox": bbox, "dataset": dataset}
    return index


def _discover_scenes(gen_dir, method, gt_index, dataset_re, done_keys):
    """Discover scene entries to evaluate. Returns list of dicts with keys:
    key, scene_dir, gt_scene_dir, scene_name, dataset, bbox."""
    entries = []

    if method in ("sgnn", "nksr"):
        # Flat-file layout: {scene}_{bbox6}pred.ply
        for ply_path in sorted(glob.glob(os.path.join(gen_dir, "*pred.ply"))):
            stem = os.path.basename(ply_path).replace("pred.ply", "")
            parts = stem.rsplit("_", 6)
            if len(parts) != 7:
                continue
            scene_name = parts[0]
            if dataset_re is not None and not dataset_re.match(scene_name):
                continue
            try:
                bbox = [float(x) for x in parts[1:]]
            except ValueError:
                continue
            key = f"{scene_name}_{_bbox_min_key(bbox)}"
            if key in done_keys:
                continue
            gt_row = gt_index.get((scene_name, _bbox_min_key(bbox)))
            if gt_row is None:
                continue
            entries.append(
                {
                    "key": key,
                    "scene_dir": ply_path,
                    "gt_scene_dir": gt_row["gt_scene_dir"],
                    "scene_name": scene_name,
                    "dataset": gt_row["dataset"],
                    "bbox": gt_row["bbox"],
                }
            )
    else:
        # Subdirectory layout: {scene_name}_{bbox_min_str}/
        scene_dirs = sorted(
            d
            for d in os.listdir(gen_dir)
            if os.path.isdir(os.path.join(gen_dir, d))
            and (dataset_re is None or dataset_re.match(d))
        )
        for dir_name in scene_dirs:
            if dir_name in done_keys:
                continue
            parts = dir_name.rsplit("_", 3)
            if len(parts) != 4:
                continue
            scene_name = parts[0]
            bbox_min_str = "_".join(parts[1:])
            gt_row = gt_index.get((scene_name, bbox_min_str))
            if gt_row is None:
                continue
            entries.append(
                {
                    "key": dir_name,
                    "scene_dir": os.path.join(gen_dir, dir_name),
                    "gt_scene_dir": gt_row["gt_scene_dir"],
                    "scene_name": scene_name,
                    "dataset": gt_row["dataset"],
                    "bbox": gt_row["bbox"],
                }
            )

    return entries


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(cfg: Config):
    # Build GT index from pre-exported gt_dir
    gt_index = _load_gt_index(cfg.gt_dir)
    print(f"Loaded {len(gt_index)} GT scenes from {cfg.gt_dir}")

    dataset_re = _DATASET_PATTERNS.get(cfg.dataset) if cfg.dataset else None

    # --resume: load already-computed scenes
    all_metrics = {}
    if cfg.resume and cfg.output:
        all_metrics = _load_partial_results(cfg.output)
        if all_metrics:
            print(f"Resumed {len(all_metrics)} scenes from {cfg.output}.partial")

    # Discover scenes (filtering out already-done ones)
    entries = _discover_scenes(
        cfg.gen_dir, cfg.method, gt_index, dataset_re, set(all_metrics.keys())
    )
    if not entries:
        if all_metrics:
            print("All scenes already computed (resume). Skipping to aggregation.")
        else:
            print(f"No scenes found in {cfg.gen_dir}")
            return

    # --export_pair: save one GT+pred mesh pair for debugging, then exit
    if cfg.export_pair is not None:
        entry = entries[0]
        out_dir = cfg.export_pair
        os.makedirs(out_dir, exist_ok=True)
        meta = load_meta(entry["scene_dir"]) if cfg.method == "ours" else None
        voxel_size = meta["voxel_size"] if meta else DEFAULT_VOXEL_SIZE
        mesh_gt = load_gt_mesh(entry["gt_scene_dir"])
        meshes_pred = load_pred_meshes(
            entry["scene_dir"], cfg.method, bbox=entry["bbox"], voxel_size=voxel_size
        )
        if mesh_gt is not None:
            gt_path = os.path.join(out_dir, f"{entry['key']}_gt.ply")
            mesh_gt.export(gt_path)
            v = mesh_gt.vertices
            print(
                f"GT mesh: {gt_path}  verts={len(v)}  range=[{v.min(0)} .. {v.max(0)}]"
            )
        if meshes_pred:
            pred_path = os.path.join(out_dir, f"{entry['key']}_pred.ply")
            meshes_pred[0].export(pred_path)
            v = meshes_pred[0].vertices
            print(
                f"Pred mesh: {pred_path}  verts={len(v)}  range=[{v.min(0)} .. {v.max(0)}]"
            )
        return

    if cfg.max_samples is not None and len(entries) > cfg.max_samples:
        entries = entries[: cfg.max_samples]
        print(f"Capped to {cfg.max_samples} scenes")

    print(
        f"Found {len(entries)} scenes to process | method: {cfg.method} | metrics: {cfg.metrics}"
    )

    # DataLoader for parallel I/O
    ds = SceneDataset(entries, cfg.method, cfg.metrics)
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=1,
        num_workers=cfg.num_workers,
        collate_fn=_passthrough_collate,
        shuffle=False,
    )

    skipped = 0
    for data in tqdm(loader, desc="Computing metrics", total=len(entries)):
        key = data["key"]
        result = compute_metrics_from_loaded(data, cfg.metrics, cfg.device)
        if result is None:
            skipped += 1
        else:
            all_metrics[key] = result
            if cfg.output:
                _save_partial_results(cfg.output, all_metrics)

    if skipped:
        print(f"Skipped {skipped} scenes (missing GT or pred data)")

    if not all_metrics:
        print("No metrics computed.")
        return

    # Aggregate with NaN-safe mean
    aggregated = {}
    for scene_metrics in all_metrics.values():
        for k, v in scene_metrics.items():
            aggregated.setdefault(k, []).append(v)

    count = len(all_metrics)
    summary = {}
    print(f"\nAggregated metrics ({count} scenes):")
    for k, vals in sorted(aggregated.items()):
        arr = np.array(vals, dtype=np.float64)
        mean_val = float(np.nanmean(arr))
        std_val = float(np.nanstd(arr))
        summary[k] = {"mean": mean_val, "std": std_val, "n": count}
        print(f"  {k}: {mean_val:.6f} ± {std_val:.6f}")

    if cfg.output:
        os.makedirs(os.path.dirname(cfg.output) or ".", exist_ok=True)
        with open(cfg.output, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Saved to {cfg.output}")
        # Clean up partial file on successful completion
        partial_path = cfg.output + ".partial"
        if os.path.exists(partial_path):
            os.remove(partial_path)
            print(f"Cleaned up {partial_path}")


def _cli() -> None:
    faulthandler.enable()
    main(tyro.cli(Config))


if __name__ == "__main__":
    _cli()
