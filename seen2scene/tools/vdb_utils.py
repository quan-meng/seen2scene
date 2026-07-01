import torch
import fvdb
import sys
import os
import trimesh
import numpy as np
import matplotlib.pyplot as plt

try:
    import pyopenvdb as vdb
except ImportError:
    import openvdb as vdb

import fvdb.nn as fvnn
from typing import *
from sklearn.decomposition import PCA

from seen2scene.models import sparse as sp


def loadvdb(vdb_path):
    """Load VDB file with retry logic if file is locked.

    Args:
        vdb_path: Path to VDB file
        retry_delay: Delay in seconds between retries
    """
    grids, metadata = vdb.readAll(vdb_path)
    grids = {x.name: x for x in grids}
    return grids, metadata


def crop_vdb(vdb_grid, bbox_min_world, shape, cell_center=True):
    """
    Convert an OpenVDB grid to a dense 3D NumPy array within the given bounding box.

    Args:
        vdb_grid: OpenVDB grid to convert
        bbox_min_world: The minimum corner coordinates of the bounding box in world space (x, y, z)
        shape: The shape of the output array (nx, ny, nz)z

    Note:
        This function is vectorized, so it is fast.

    Returns:
        np.ndarray: Dense 3D NumPy array of shape `shape` containing the cropped volume data
    """

    if vdb_grid is None:
        raise ValueError("Invalid VDB grid")
    if not hasattr(vdb_grid, "transform"):
        raise ValueError("VDB grid missing transform")

    try:
        if cell_center:
            bbox_min = vdb_grid.transform.worldToIndexCellCentered(bbox_min_world)
        else:
            bbox_min = vdb_grid.transform.worldToIndex(bbox_min_world)

        array = np.zeros(shape, dtype=np.float32)

        vdb_grid.copyToArray(array, ijk=bbox_min)
    except Exception as e:
        raise RuntimeError(f"VDB copyToArray failed: {str(e)}")

    array = np.ascontiguousarray(array)  # ensure C-contiguous

    return array


def vdbtensor_from_coords(
    ijks: List[torch.Tensor],
    values: List[torch.Tensor],
    voxel_sizes: List[float],
    origins: List[float],
):
    # Convert coords to JaggedTensor and gridbatch
    ijks = fvdb.JaggedTensor(ijks)
    values = fvdb.JaggedTensor(values)

    grid = fvdb.gridbatch_from_ijk(ijks, voxel_sizes=voxel_sizes, origins=origins)

    # Get indexes and ensure they are valid
    indexes_jagged = grid.ijk_to_inv_index(ijks)

    # Fix: Replace assertion with runtime check (not disabled with -O flag)
    if len(indexes_jagged.jdata) > 0:
        if not (
            torch.all(indexes_jagged.jdata >= 0)
            and torch.all(indexes_jagged.jdata < len(values.jdata))
        ):
            idx_min = indexes_jagged.jdata.min().item()
            idx_max = indexes_jagged.jdata.max().item()
            raise IndexError(
                f"Index out of bounds: got indices in range [{idx_min}, {idx_max}], "
                f"expected [0, {len(values.jdata)-1}]"
            )

    values = values[indexes_jagged]

    return fvnn.VDBTensor(grid, values)


def randn_like(x: fvnn.VDBTensor):
    return fvnn.VDBTensor(
        x.grid, x.grid.jagged_like(torch.randn_like(x.data.jdata)), x.kmap
    )


def cat(x: fvnn.VDBTensor, y: fvnn.VDBTensor, dim=-1):
    assert dim == -1, "Only support concatenating along the last dimension"
    return fvnn.VDBTensor(
        x.grid,
        x.grid.jagged_like(torch.cat([x.data.jdata, y.data.jdata], dim=dim)),
    )


def stack(x: fvnn.VDBTensor, y: fvnn.VDBTensor):
    assert torch.allclose(
        x.grid.voxel_sizes, y.grid.voxel_sizes
    ), "Voxel sizes must be the same"
    assert torch.allclose(x.grid.origins, y.grid.origins), "Origins must be the same"

    voxel_sizes = torch.cat((x.grid.voxel_sizes, y.grid.voxel_sizes))
    origins = torch.cat((x.grid.origins, y.grid.origins))

    coords, values = [], []
    for batch_idx in range(len(x) + len(y)):
        if batch_idx < len(x):
            coords_i = x.grid[batch_idx].ijk.jdata
            values_i = x.data[batch_idx].jdata
        else:
            coords_i = y.grid[batch_idx - len(x)].ijk.jdata
            values_i = y.data[batch_idx - len(x)].jdata

        coords.append(coords_i)
        values.append(values_i)

    return vdbtensor_from_coords(coords, values, voxel_sizes, origins)


def split(x: fvnn.VDBTensor, split_size_or_sections: Union[int, list[int]]):
    n_batch = len(x)
    if isinstance(split_size_or_sections, int):
        chunk_size = [n_batch // split_size_or_sections] * split_size_or_sections
    else:
        chunk_size = split_size_or_sections

    assert sum(chunk_size) == n_batch

    # TODO


def vdb_to_spconv(x: fvnn.VDBTensor) -> sp.SparseTensor:
    coords = torch.cat([x.grid.jidx[:, None], x.grid.ijk.jdata], dim=-1)  # [N, 4]
    feats = x.data.jdata
    layout = [
        slice(x.data.joffsets[i], x.data.joffsets[i + 1]) for i in range(len(x.data))
    ]
    xyzs = x.grid.grid_to_world(x.grid.ijk.float()).jdata  # [N, 3]
    shape = torch.Size([len(x.grid), feats.shape[-1]])  # [N, C]
    return sp.SparseTensor(
        feats=feats, coords=coords, shape=shape, layout=layout, xyzs=xyzs
    )


def spconv_to_vdb(
    x: sp.SparseTensor,
    grid: fvdb.GridBatch,
    kmap: Optional[fvdb.SparseConvPackInfo] = None,
) -> fvnn.VDBTensor:
    jagged = grid.jagged_like(x.feats)
    return fvnn.VDBTensor(grid, jagged, kmap)


def vis_grid_as_pointclouds(
    grid: fvdb.GridBatch,
    bbox_world: torch.Tensor,
    scene_names: List[str],
    out_dir: str,
    sparsity_ratio: float = 1.0,
):
    for grid_i, bbox_world_i, scene_name_i in zip(grid, bbox_world, scene_names):
        bbox_str = "_".join([f"{x:.01f}" for x in bbox_world_i[:3].cpu().numpy()])
        out_dir_i = os.path.join(out_dir, f"{scene_name_i}_{bbox_str}")
        os.makedirs(out_dir_i, exist_ok=True)

        xyzs_i = grid_i.grid_to_world(grid_i.ijk.float()).jdata  # [N, 3]
        xyzs_i = xyzs_i + bbox_world_i[None, :3]
        mesh = trimesh.PointCloud(vertices=xyzs_i.cpu().numpy())
        mesh.export(os.path.join(out_dir_i, f"grid_sparse_{sparsity_ratio}.ply"))


def export_latent_pca(
    latent_vdb: fvnn.VDBTensor,
    bbox_world: torch.Tensor,
    scene_names: List[str],
    out_dir: str,
):
    """Apply PCA to latent features and export as RGB-colored point clouds.

    Args:
        latent_vdb: VDBTensor with per-voxel latent features.
        bbox_world: [B, 6] tensor of world-space bounding boxes.
        scene_names: List of scene name strings.
        out_dir: Root output directory.
    """
    for i, scene_name in enumerate(scene_names):
        grid_i = latent_vdb.grid[i]
        feats_i = latent_vdb.data[i].jdata  # [N, C]
        xyzs_i = grid_i.grid_to_world(grid_i.ijk.float()).jdata  # [N, 3]
        xyzs_i = xyzs_i + bbox_world[i, :3].to(xyzs_i.device)

        feats_np = feats_i.float().cpu().numpy()
        xyzs_np = xyzs_i.float().cpu().numpy()

        pca = PCA(n_components=3)
        pca_feats = pca.fit_transform(feats_np)  # [N, 3]

        for c in range(3):
            col = pca_feats[:, c]
            cmin, cmax = col.min(), col.max()
            if cmax - cmin > 1e-8:
                pca_feats[:, c] = (col - cmin) / (cmax - cmin)
            else:
                pca_feats[:, c] = 0.5
        colors = (pca_feats * 255).astype(np.uint8)

        bbox_str = "_".join([f"{x:.01f}" for x in bbox_world[i, :3].cpu().numpy()])
        out_dir_i = os.path.join(out_dir, f"{scene_name}_{bbox_str}")
        os.makedirs(out_dir_i, exist_ok=True)
        pc = trimesh.PointCloud(vertices=xyzs_np, colors=colors)
        pc.export(os.path.join(out_dir_i, "latent_pca.ply"))


def export_scan_prob_heatmap(
    struct_vdb: fvnn.VDBTensor,
    bbox_world: torch.Tensor,
    scene_names: List[str],
    out_dir: str,
    ins_mask: Optional[torch.Tensor] = None,
):
    """Export scan probability as a heatmap-colored point cloud.

    Computes the scan probability using neighbor averaging of structure values,
    matching the computation in ``Encoder.structure_encoding``:
    - ``neighbor_indexes(ijk, 2)`` → 5x5x5 neighborhood
    - EMPTY → 1.0, BAND → 0.5, else → 0.0
    - Average over neighbors → per-voxel scan probability

    Args:
        struct_vdb: VDBTensor with per-voxel structure values
            (EMPTY=0, UNKNOWN=1, BAND=2), as produced by max_pool3d on the
            dense structure tensor followed by vdbtensor_from_dense.
        bbox_world: [B, 6] tensor of world-space bounding boxes.
        scene_names: List of scene name strings.
        out_dir: Root output directory.
        ins_mask: Optional [N_total, M] instance mask from get_instance_mask.
            When provided, never_scan masks are filtered to only include
            voxels inside object bboxes (ins_mask.any(dim=-1)).
    """
    import matplotlib.cm as cm
    from seen2scene.configs.dataset import Voxel

    # Compute neighbor-based scan probability (same as encoder.structure_encoding)
    neighbor_idxs = struct_vdb.grid.neighbor_indexes(
        struct_vdb.grid.ijk, 2
    )  # [N, 5, 5, 5]
    neighbor_values = struct_vdb.data.jdata[neighbor_idxs.jdata]  # [N, 5, 5, 5, C]
    neighbor_values = torch.where(
        neighbor_values == Voxel.EMPTY,
        1.0,
        torch.where(neighbor_values == Voxel.BAND, 0.5, 0.0),
    )
    scan_prob_all = neighbor_values.mean(dim=[-1, -2, -3, -4])  # [N_total]

    # Per-voxel object membership mask (flattened across batch)
    obj_mask = ins_mask.any(dim=-1).cpu().numpy() if ins_mask is not None else None
    offset = 0

    for i, scene_name in enumerate(scene_names):
        grid_i = struct_vdb.grid[i]
        n_i = grid_i.num_voxels
        vals_i = scan_prob_all[offset : offset + n_i].float().cpu().numpy()  # [N]
        xyzs_i = grid_i.grid_to_world(grid_i.ijk.float()).jdata  # [N, 3]
        xyzs_i = (xyzs_i + bbox_world[i, :3].to(xyzs_i.device)).float().cpu().numpy()

        obj_mask_i = obj_mask[offset : offset + n_i] if obj_mask is not None else None
        offset += n_i

        colors = (cm.jet(vals_i)[:, :3] * 255).astype(np.uint8)

        bbox_str = "_".join([f"{x:.01f}" for x in bbox_world[i, :3].cpu().numpy()])
        out_dir_i = os.path.join(out_dir, f"{scene_name}_{bbox_str}")
        os.makedirs(out_dir_i, exist_ok=True)
        pc = trimesh.PointCloud(vertices=xyzs_i, colors=colors)
        pc.export(os.path.join(out_dir_i, "scan_prob_heatmap.ply"))

        # Export never_scan_mask at different thresholds
        for thresh in (0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5):
            mask = vals_i < thresh
            if obj_mask_i is not None:
                mask = mask & obj_mask_i
            if mask.sum() == 0:
                continue
            pc_mask = trimesh.PointCloud(
                vertices=xyzs_i[mask], colors=colors[mask]
            )
            pc_mask.export(
                os.path.join(out_dir_i, f"never_scan_mask_thresh_{thresh:.1f}.ply")
            )


def export_rope_pca(
    latent_vdb: fvnn.VDBTensor,
    conditioner,
    object_names_batch: List[List[str]],
    object_bboxes_batch: List[List[torch.Tensor]],
    bbox_world: torch.Tensor,
    scene_names: List[str],
    out_dir: str,
    use_distance_rope: bool = True,
):
    """Apply PCA to CLIP embeddings with distance RoPE and export as RGB-colored point clouds.

    Args:
        latent_vdb: VDBTensor with per-voxel latent features.
        conditioner: Conditioner module for encoding.
        object_names_batch: List of object names per scene.
        object_bboxes_batch: List of object bounding boxes per scene.
        bbox_world: [B, 6] tensor of world-space bounding boxes.
        scene_names: List of scene name strings.
        out_dir: Root output directory.
        use_distance_rope: Whether to use distance RoPE conditioning.
    """
    from seen2scene.models.trainer.common import get_instance_mask
    from sklearn.decomposition import PCA

    # Create instance mask
    xyzs = latent_vdb.grid.grid_to_world(latent_vdb.grid.ijk.float())  # [N, 3]
    xyzs_list = [xyzs_i.jdata for xyzs_i in xyzs]  # List of [N_i, 3] tensors
    ins_mask = get_instance_mask(xyzs_list, object_bboxes_batch)

    # Create mock latent for conditioner
    device = latent_vdb.device
    N_total = sum(len(latent_vdb.grid[i].ijk) for i in range(len(latent_vdb.grid)))

    class MockLatent:
        def __init__(self, coords, feats, xyzs, device):
            self.coords = coords
            self.feats = feats
            self.xyzs = xyzs
            self.device = device

        def replace(self, new_feats):
            return MockLatent(self.coords, new_feats, self.xyzs, self.device)

        def to(self, device):
            return self

    # Extract coordinates
    voxel_coords_list = []
    coords_list = []
    offset = 0

    for i in range(len(latent_vdb.grid)):
        grid_i = latent_vdb.grid[i]

        # Get actual voxel coordinates
        ijk_data = grid_i.ijk.jdata  # [K, 3]
        N_i = len(ijk_data)

        # World coordinates
        xyzs_i = grid_i.grid_to_world(grid_i.ijk.float()).jdata
        xyzs_i = xyzs_i + bbox_world[i, :3].to(xyzs_i.device)
        voxel_coords_list.append(xyzs_i)

        # Grid coordinates for mock latent
        coords_i = torch.zeros(N_i, 4, device=device)
        coords_i[:, 0] = i  # Batch index
        coords_i[:, 1:] = ijk_data.float()
        coords_list.append(coords_i)

    voxel_coords = torch.cat(voxel_coords_list, dim=0)
    coords = torch.cat(coords_list, dim=0)
    feats = torch.randn(N_total, 256, device=device)  # Placeholder features

    latent_mock = MockLatent(coords, feats, voxel_coords, device)

    # Run through conditioner
    with torch.no_grad():
        result = conditioner(
            latent_mock,
            object_names_batch,
            ins_mask,
            object_bboxes_batch if use_distance_rope else None,
        )

    # Split by scene and export
    offset = 0
    for i, scene_name in enumerate(scene_names):
        N_i = len(latent_vdb.grid[i].ijk)

        # Extract features for this scene
        feats_i = result.feats[offset : offset + N_i]
        xyzs_i = voxel_coords[offset : offset + N_i]
        offset += N_i

        # Apply PCA
        feats_np = feats_i.float().cpu().numpy()
        xyzs_np = xyzs_i.float().cpu().numpy()

        if len(feats_np) < 3:
            print(f"Skipping {scene_name}: too few points ({len(feats_np)})")
            continue

        pca = PCA(n_components=3)
        pca_feats = pca.fit_transform(feats_np)

        # Normalize to [0, 1]
        for c in range(3):
            col = pca_feats[:, c]
            cmin, cmax = col.min(), col.max()
            if cmax - cmin > 1e-8:
                pca_feats[:, c] = (col - cmin) / (cmax - cmin)
            else:
                pca_feats[:, c] = 0.5

        colors = (pca_feats * 255).astype(np.uint8)

        # Export
        bbox_str = "_".join([f"{x:.01f}" for x in bbox_world[i, :3].cpu().numpy()])
        out_dir_i = os.path.join(out_dir, f"{scene_name}_{bbox_str}")
        os.makedirs(out_dir_i, exist_ok=True)

        filename = f"rope_pca_{'with' if use_distance_rope else 'without'}_rope.ply"
        pc = trimesh.PointCloud(vertices=xyzs_np, colors=colors)
        pc.export(os.path.join(out_dir_i, filename))

        print(f"Exported RoPE PCA visualization: {scene_name}/{filename}")


def export_rope_pca_visualizations(
    latent_sp,
    context_with_rope,
    context_without_rope,
    scene_names: Sequence[str],
    out_dir: str,
    object_names_batch=None,
    object_bboxes_batch=None,
    ins_mask=None,
    bbox_world=None,
) -> None:
    """Per-object visualization showing spatial gradients and RoPE effects.

    Args:
        latent_sp: Sparse tensor with coords and xyzs for voxel positions.
        context_with_rope: Conditioner output with distance RoPE enabled.
        context_without_rope: Conditioner output with distance RoPE disabled.
        scene_names: List of scene name strings.
        out_dir: Root output directory.
        object_names_batch: List of object names per scene.
        object_bboxes_batch: List of object bboxes per scene.
        ins_mask: Instance mask [N, L] indicating voxel-object associations.
        bbox_world: Bounding box offsets [B, 6] to convert to world coordinates.
    """
    import matplotlib.pyplot as plt
    import torch

    for i, scene_name in enumerate(scene_names):
        # Get voxel coordinates for this scene
        scene_mask = latent_sp.coords[:, 0] == i
        xyzs_i = latent_sp.xyzs[scene_mask]

        # Get features for this scene (plain tensor or SparseTensor)
        wr_feats = context_with_rope.feats if hasattr(context_with_rope, 'feats') else context_with_rope
        wor_feats = context_without_rope.feats if hasattr(context_without_rope, 'feats') else context_without_rope
        feats_with_rope = wr_feats[scene_mask]
        feats_without_rope = wor_feats[scene_mask]

        if len(feats_with_rope) < 3:
            print(f"Skipping {scene_name}: too few points ({len(feats_with_rope)})")
            continue

        # Convert to world coordinates by adding bbox_world offset
        xyzs_np = xyzs_i.float().cpu().numpy()
        world_offset = bbox_world[i, :3].cpu().numpy()
        xyzs_np = xyzs_np + world_offset
        bbox_str = "_".join([f"{x:.01f}" for x in bbox_world[i, :3].cpu().numpy()])
        out_scene_dir = os.path.join(out_dir, f"{scene_name}_{bbox_str}")
        os.makedirs(out_scene_dir, exist_ok=True)

        # Get instance mask and object info for this scene
        if ins_mask is not None and object_names_batch is not None:
            ins_mask_i = ins_mask[scene_mask]

            # Count objects per batch up to scene i
            obj_offset = sum(len(names) for names in object_names_batch[:i])
            num_objects_i = len(object_names_batch[i])

            # Extract relevant columns for this scene's objects
            ins_mask_i = ins_mask_i[:, obj_offset : obj_offset + num_objects_i]
            object_names_i = object_names_batch[i]
            object_bboxes_i = object_bboxes_batch[i]
        else:
            ins_mask_i = None
            object_names_i = None
            object_bboxes_i = None

        # Compute difference magnitude
        diff_feats = feats_with_rope - feats_without_rope
        diff_magnitude = torch.norm(diff_feats, dim=1).cpu().numpy()

        # OPTION 5: PER-OBJECT VISUALIZATION
        if (
            ins_mask_i is not None
            and object_bboxes_i is not None
            and len(object_bboxes_i) > 0
        ):

            # 1. Compute NORMALIZED distances (same as flow_matching.py) for voxels INSIDE bboxes
            # Compute bbox centers and sizes
            bbox_centers = []
            bbox_sizes = []
            for obj_bbox in object_bboxes_i:
                center = (obj_bbox[0] + obj_bbox[1]) / 2
                size = obj_bbox[1] - obj_bbox[0]
                bbox_centers.append(center)
                bbox_sizes.append(size)
            bbox_centers = torch.stack(bbox_centers)  # [num_objects, 3]
            bbox_sizes = torch.stack(bbox_sizes)  # [num_objects, 3]

            # For each voxel with an object, compute normalized distance
            distances_to_nearest = np.full(len(xyzs_i), np.nan)
            nearest_object_idx = np.full(len(xyzs_i), -1, dtype=int)

            for voxel_idx in range(len(xyzs_i)):
                voxel_xyz = xyzs_i[voxel_idx]
                min_normalized_dist = float("inf")
                nearest_idx = -1

                # Check all objects this voxel belongs to (ins_mask > 0)
                for obj_idx in range(len(object_bboxes_i)):
                    if not ins_mask_i[voxel_idx, obj_idx]:
                        continue  # Skip if voxel doesn't belong to this object

                    # Compute NORMALIZED distance (same as flow_matching.py line 341-342)
                    rel_pos = voxel_xyz - bbox_centers[obj_idx].to(voxel_xyz.device)
                    normalized_rel_pos = (
                        rel_pos
                        * 2.0
                        / (bbox_sizes[obj_idx].to(voxel_xyz.device) + 1e-6)
                    )
                    normalized_dist = torch.norm(normalized_rel_pos).item()

                    if normalized_dist < min_normalized_dist:
                        min_normalized_dist = normalized_dist
                        nearest_idx = obj_idx

                if nearest_idx >= 0:
                    distances_to_nearest[voxel_idx] = min_normalized_dist
                    nearest_object_idx[voxel_idx] = nearest_idx

            # Filter to only valid voxels (inside objects)
            valid_mask = ~np.isnan(distances_to_nearest)
            if valid_mask.sum() == 0:
                print(f"  ⚠ No voxels inside object bounding boxes")
                ins_mask_i = None  # Skip visualization

            # 2. DISTANCE TO OBJECT visualization (only voxels inside object bboxes)
            if valid_mask.sum() > 0 and np.nanmax(distances_to_nearest) > 1e-8:
                valid_distances = distances_to_nearest[valid_mask]
                valid_xyzs = xyzs_np[valid_mask]

                norm_dist = valid_distances / valid_distances.max()
                colors = plt.cm.viridis(norm_dist)[:, :3] * 255
                colors = colors.astype(np.uint8)

                out_path = os.path.join(out_scene_dir, "rope_distance_to_object.ply")
                pc = trimesh.PointCloud(vertices=valid_xyzs, colors=colors)
                pc.export(out_path)
                print(
                    f"✓ Exported normalized distance visualization: {scene_name}/rope_distance_to_object.ply"
                )
                print(
                    f"  Normalized distance range: {valid_distances.min():.3f} - {valid_distances.max():.3f}"
                )
                print(f"  Voxels inside bboxes: {valid_mask.sum()} / {len(xyzs_np)}")

            # 3. DIFFERENCE MAGNITUDE visualization (blue=no change, red=large change)
            if diff_magnitude.max() > 1e-8:
                norm_diff = diff_magnitude / diff_magnitude.max()
                colors = plt.cm.coolwarm(norm_diff)[:, :3] * 255
                colors = colors.astype(np.uint8)

                out_path = os.path.join(out_scene_dir, "rope_difference_magnitude.ply")
                pc = trimesh.PointCloud(vertices=xyzs_np, colors=colors)
                pc.export(out_path)
                print(
                    f"✓ Exported difference magnitude: {scene_name}/rope_difference_magnitude.ply"
                )
                print(
                    f"  Diff range: {diff_magnitude.min():.6f} - {diff_magnitude.max():.6f}"
                )

                # Compute correlation between distance and difference magnitude
                # (Should be positive if RoPE has spatial effect) - only for valid voxels
                if valid_mask.sum() > 1:
                    valid_diff = diff_magnitude[valid_mask]
                    correlation = np.corrcoef(valid_distances, valid_diff)[0, 1]
                    print(
                        f"  📊 Correlation(normalized_distance, diff_magnitude): {correlation:.4f}"
                    )
                    if abs(correlation) > 0.3:
                        print(
                            f"     → {'Strong' if abs(correlation) > 0.6 else 'Moderate'} spatial pattern detected!"
                        )

            # 4. PER-OBJECT exports (show gradient within each object's region)
            print(f"\n  Per-object visualizations for {len(object_names_i)} objects:")
            for obj_idx, obj_name in enumerate(object_names_i):
                obj_mask = ins_mask_i[:, obj_idx] > 0
                if obj_mask.sum() < 3:
                    continue

                obj_mask_np = obj_mask.cpu().numpy()
                obj_xyzs_np = xyzs_np[obj_mask_np]  # Use numpy array with numpy mask
                obj_diff = diff_magnitude[obj_mask_np]
                obj_bbox = object_bboxes_i[obj_idx]
                obj_center = ((obj_bbox[0] + obj_bbox[1]) / 2).cpu().numpy()

                # Compute distance from each voxel to THIS object's center
                obj_distances = np.linalg.norm(obj_xyzs_np - obj_center, axis=1)

                # Color by difference magnitude within this object
                if obj_diff.max() > 1e-8:
                    obj_norm_diff = obj_diff / obj_diff.max()
                    obj_colors = plt.cm.coolwarm(obj_norm_diff)[:, :3] * 255
                    obj_colors = obj_colors.astype(np.uint8)

                    # Sanitize object name for filename
                    safe_name = "".join(
                        c if c.isalnum() or c in "_ -" else "_" for c in obj_name
                    )
                    out_path = os.path.join(
                        out_scene_dir, f"object_{obj_idx:02d}_{safe_name}_diff.ply"
                    )
                    pc = trimesh.PointCloud(vertices=obj_xyzs_np, colors=obj_colors)
                    pc.export(out_path)

                    # Compute correlation for this object
                    obj_corr = (
                        np.corrcoef(obj_distances, obj_diff)[0, 1]
                        if len(obj_distances) > 1
                        else 0.0
                    )
                    print(
                        f"    [{obj_idx}] {obj_name}: {obj_mask.sum()} voxels, corr={obj_corr:.3f}"
                    )

        else:
            print(f"  ⚠ Skipping per-object visualization: no object info available")

            # Fallback: just show difference magnitude
            if diff_magnitude.max() > 1e-8:
                norm_diff = diff_magnitude / diff_magnitude.max()
                colors = plt.cm.coolwarm(norm_diff)[:, :3] * 255
                colors = colors.astype(np.uint8)

                out_path = os.path.join(out_scene_dir, "rope_difference_magnitude.ply")
                pc = trimesh.PointCloud(vertices=xyzs_np, colors=colors)
                pc.export(out_path)
                print(
                    f"✓ Exported difference magnitude (no objects): {scene_name}/rope_difference_magnitude.ply"
                )
