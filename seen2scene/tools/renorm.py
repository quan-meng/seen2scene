"""Coordinate renormalization for baseline methods.

All baseline methods produce meshes in different coordinate systems.
These functions transform them to our Z-up world coordinate system (metres).

Every renorm function accepts a ``ceiling_clip`` parameter (default 2.2 m,
matching ``Dataset.ceiling_clip``).  The mesh is clipped at
``z_min + ceiling_clip``, but the returned bbox retains the full patch
height (``z_max = z_min + PATCH_SIZE``).  This keeps camera placement
consistent across samples.  Pass ``ceiling_clip=None`` to disable clipping.
"""

import numpy as np
import trimesh

from seen2scene.tools.common_utils import clip_mesh

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

PATCH_SIZE = 2.8  # Our patch cube side length (metres)
CEILING_CLIP = 2.2  # Default ceiling clip height (must match Dataset.ceiling_clip)
BLOCKFUSION_SIZE = 2.8  # BlockFusion canonical cube maps to 2.8 m in world
WORLDGROW_SCALE = 3.0  # WorldGrow: 1 normalized unit ≈ 3 m
# LT3SD: 256×128×256 Y-up → [256, 256, 128] Z-up → [5.6, 5.6, 2.8] m
LT3SD_FULL_BBOX = [0.0, 0.0, 0.0, 2 * PATCH_SIZE, 2 * PATCH_SIZE, PATCH_SIZE]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def apply_ceiling_clip(bbox, ceiling_clip=CEILING_CLIP):
    """Override bbox z_max to z_min + ceiling_clip (same logic as render pipeline)."""
    bbox = list(bbox)
    if ceiling_clip is not None:
        bbox[5] = bbox[2] + ceiling_clip
    return bbox


def split_bbox_xy(bbox):
    """Split bbox into 4 quadrants at XY midpoint (Z stays full)."""
    xmin, ymin, zmin, xmax, ymax, zmax = bbox
    xmid = (xmin + xmax) / 2
    ymid = (ymin + ymax) / 2
    return [
        [xmin, ymin, zmin, xmid, ymid, zmax],
        [xmid, ymin, zmin, xmax, ymid, zmax],
        [xmin, ymid, zmin, xmid, ymax, zmax],
        [xmid, ymid, zmin, xmax, ymax, zmax],
    ]


def parse_bbox_from_stem(stem):
    """Parse '{scene_name}_{x:.1f}_{y:.1f}_{z:.1f}_{x2:.1f}_{y2:.1f}_{z2:.1f}'.

    Returns (scene_name, bbox_list) or None if the last 6 tokens are not floats.
    """
    parts = stem.rsplit("_", 6)
    if len(parts) < 7:
        return None
    try:
        bbox = [float(p) for p in parts[-6:]]
    except ValueError:
        return None
    return "_".join(parts[:-6]), bbox


# ---------------------------------------------------------------------------
# Renormalization functions
# ---------------------------------------------------------------------------


def renorm_blockfusion(mesh, bbox, ceiling_clip=CEILING_CLIP):
    """BlockFusion: [-1,1] Y-up → Z-up world metres.

    Args:
        mesh:  Trimesh loaded from BlockFusion PLY.
        bbox:  6-float bbox from CSV or filename.  Only bbox[:3] (min corner) is used.
        ceiling_clip: Clip mesh at z_min + ceiling_clip.  None to disable.

    Returns (mesh, render_bbox):
        mesh:        Vertices transformed and clipped at ceiling height.
        render_bbox: [bbox_min, bbox_min + PATCH_SIZE] (full patch, not clipped).
    """
    mesh.vertices[:, [1, 2]] = mesh.vertices[:, [2, 1]]  # Y↔Z swap
    mins = np.array(bbox[:3])
    mesh.vertices = (mesh.vertices + 1.0) / 2.0 * BLOCKFUSION_SIZE + mins
    render_bbox = list(mins) + [
        mins[0] + PATCH_SIZE,
        mins[1] + PATCH_SIZE,
        mins[2] + PATCH_SIZE,
    ]
    if ceiling_clip is not None:
        clip_box = apply_ceiling_clip(list(render_bbox), ceiling_clip)
        clipped = clip_mesh(mesh, clip_box)
        if clipped is not None:
            mesh = clipped
    return mesh, render_bbox


def renorm_lt3sd(mesh, bbox, ceiling_clip=CEILING_CLIP):
    """LT3SD (scene completion): [-1,1] Y-up → Z-up world metres using per-patch bbox.

    Returns (mesh, bbox) — mesh clipped at ceiling, bbox retains full height.
    """
    mesh.vertices[:, [1, 2]] = mesh.vertices[:, [2, 1]]  # Y↔Z swap
    voxel = (mesh.vertices + 1.0) / 2.0 * 127.0
    dims = np.array([256.0, 256.0, 128.0])
    bbox_min = np.array(bbox[:3])
    bbox_max = np.array(bbox[3:])
    mesh.vertices = voxel / (dims - 1.0) * (bbox_max - bbox_min) + bbox_min
    if ceiling_clip is not None:
        clip_box = apply_ceiling_clip(list(bbox), ceiling_clip)
        clipped = clip_mesh(mesh, clip_box)
        if clipped is not None:
            mesh = clipped
    return mesh, list(bbox)


def renorm_lt3sd_sg(mesh, ceiling_clip=CEILING_CLIP):
    """LT3SD (scene generation): [-1,1] Y-up → Z-up world metres, split into 4 quadrants.

    Returns list of (sub_mesh, sub_bbox) for each non-empty quadrant.
    Meshes are clipped at ceiling; sub_bbox retains full quadrant height.
    """
    mesh.vertices[:, [1, 2]] = mesh.vertices[:, [2, 1]]  # Y↔Z swap
    voxel = (mesh.vertices + 1.0) / 2.0 * 127.0
    dims = np.array([256.0, 256.0, 128.0])
    bbox_min = np.array(LT3SD_FULL_BBOX[:3])
    bbox_max = np.array(LT3SD_FULL_BBOX[3:])
    mesh.vertices = voxel / (dims - 1.0) * (bbox_max - bbox_min) + bbox_min
    results = []
    for sub_bbox in split_bbox_xy(LT3SD_FULL_BBOX):
        clip_box = apply_ceiling_clip(list(sub_bbox), ceiling_clip)
        sub_mesh = clip_mesh(mesh, clip_box)
        if sub_mesh is not None and len(sub_mesh.vertices) > 0:
            results.append((sub_mesh, sub_bbox))
    return results


def renorm_worldgrow(mesh, ceiling_clip=CEILING_CLIP):
    """WorldGrow: [-0.5, 0.5]³ Y-up → Z-up world metres.

    WorldGrow PLYs (world_size=1×1) have vertices in [-0.5, 0.5]³ Y-up.
    We swap Y↔Z, scale by WORLDGROW_SCALE (3 m), and shift +0.5 so the
    block lands in [0, WORLDGROW_SCALE]³.  The bbox is deterministic —
    no vertex-dependent computation.

    Returns (mesh, full_bbox) or None if empty after crop.
    Mesh is clipped at ceiling; full_bbox retains full block height.
    """
    mesh.vertices[:, [1, 2]] = mesh.vertices[:, [2, 1]]  # Y↔Z swap
    mesh.vertices = (mesh.vertices + 0.5) * WORLDGROW_SCALE  # [-0.5,0.5] → [0,3]
    full_bbox = [0.0, 0.0, 0.0, WORLDGROW_SCALE, WORLDGROW_SCALE, WORLDGROW_SCALE]
    clip_box = apply_ceiling_clip(list(full_bbox), ceiling_clip)
    mesh = clip_mesh(mesh, clip_box)
    if mesh is None or len(mesh.vertices) == 0:
        return None
    return mesh, full_bbox
