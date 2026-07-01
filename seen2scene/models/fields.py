import torch
import copy
import os
import trimesh
import fvdb
import numpy as np
import fvdb.nn as fvnn
import torch.nn.functional as F
from typing import *
import json

from seen2scene.tools.vis_utils import (
    plot_object_bboxes,
    load_semantic_palette,
    get_clip_color,
)
from seen2scene.configs.dataset import Voxel
import seen2scene.tools.mesh_utils as tc
from seen2scene.tools.common_utils import get_patch_shape, bbox2str
from seen2scene.tools.log_utils import get_logger


logger = get_logger(__name__, debug="fields")


def split_kwargs(
    scene_kwargs: Dict[str, Any],
    patch_shape: Tuple[int, int, int],
    overlap: float,
    factor: int = 1,
) -> Dict[str, Any]:
    """Split scene kwargs into batched patch kwargs.

    Computes a grid of overlapping patches covering the scene volume, assigns
    intersecting objects to each patch, and returns a single dict where
    per-patch fields have batch dimension N and scalar fields stay as-is.

    Args:
        scene_kwargs: Scene-level kwargs dict with keys: bbox_world [1, 6],
            voxel_size, object_names, object_bboxes, etc.
        patch_shape: Target patch shape in voxels (Px, Py, Pz).
        overlap: Overlap ratio (0-1) between adjacent patches.
        factor: VAE downsampling factor. When > 1, patch start positions are
            aligned to multiples of this factor to prevent latent-to-voxel
            coordinate misalignment.

    Returns:
        Dict with the same keys as scene_kwargs. Per-patch fields are expanded
        to batch dimension N (number of patches):
            - bbox_world: [N, 6] tensor
            - object_names: [[str], ...] list of N lists
            - object_bboxes: [Tensor, ...] list of N tensors [M_i, 2, 3]
        Scalar fields (voxel_size, known_ratio, etc.) stay unchanged.
    """
    bbox_world = scene_kwargs["bbox_world"]
    voxel_size = scene_kwargs["voxel_size"]
    obj_bboxes = scene_kwargs["object_bboxes"][0]  # [M, 2, 3]
    obj_names = scene_kwargs["object_names"][0]  # List[str]
    device = bbox_world.device

    scene_shape = torch.tensor(
        get_patch_shape(bbox_world, voxel_size), device=device, dtype=torch.long
    )
    patch_shape_t = torch.tensor(patch_shape, device=device, dtype=torch.long)

    overlap_shape = torch.round(patch_shape_t.float() * overlap).long()
    stride_shape = patch_shape_t - overlap_shape

    # Align stride to multiples of factor so that latent-space and voxel-space
    # patch boundaries coincide exactly (stride 205 // 4 * 4 = 204).
    if factor > 1:
        stride_shape = (stride_shape // factor) * factor
        stride_shape = torch.clamp(stride_shape, min=factor)

    patch_size = patch_shape_t.float() * voxel_size

    starts = []
    for i in range(3):
        scene_dim = scene_shape[i].item()
        patch_dim = patch_shape_t[i].item()
        stride_dim = stride_shape[i].item()
        if scene_dim <= patch_dim:
            starts.append([0])
            continue
        last = scene_dim - patch_dim
        # Clamp last start so the final patch ends exactly at the structure
        # boundary.  Earlier code rounded UP to a factor multiple, but that
        # pushed the patch past scene_dim, causing size mismatches in
        # split_structure / split_fvdb.
        if factor > 1:
            last = (last // factor) * factor
        axis = torch.arange(0, last + 1, stride_dim, device=device, dtype=torch.long)
        if axis.numel() == 0 or axis[-1].item() != last:
            axis = torch.cat(
                [axis, torch.tensor([last], device=device, dtype=axis.dtype)]
            )
        starts.append(axis.tolist())

    start_x = torch.tensor(starts[0], device=device, dtype=torch.long)
    start_y = torch.tensor(starts[1], device=device, dtype=torch.long)
    start_z = torch.tensor(starts[2], device=device, dtype=torch.long)
    grid_x, grid_y, grid_z = torch.meshgrid(start_x, start_y, start_z, indexing="ij")
    start_voxels = torch.stack((grid_x, grid_y, grid_z), dim=-1).reshape(-1, 3)
    patch_min = start_voxels * voxel_size
    patch_max = patch_min + patch_size
    patch_bbox_world = torch.cat(
        [bbox_world[0, :3] + patch_min, bbox_world[0, :3] + patch_max],
        dim=1,
    )  # [N, 6]

    num_patches = patch_min.shape[0]
    if obj_bboxes is None or len(obj_bboxes) == 0:
        empty_boxes = torch.zeros((0, 2, 3), device=device, dtype=patch_min.dtype)
        names_list = [[] for _ in range(num_patches)]
        boxes_list = [empty_boxes.clone() for _ in range(num_patches)]
    else:
        obj_min = obj_bboxes[:, 0]
        obj_max = obj_bboxes[:, 1]
        intersects = (
            (patch_min[:, None, 0] <= obj_max[None, :, 0])
            & (patch_max[:, None, 0] >= obj_min[None, :, 0])
            & (patch_min[:, None, 1] <= obj_max[None, :, 1])
            & (patch_max[:, None, 1] >= obj_min[None, :, 1])
            & (patch_min[:, None, 2] <= obj_max[None, :, 2])
            & (patch_max[:, None, 2] >= obj_min[None, :, 2])
        )

        names_list = []
        boxes_list = []
        for i in range(num_patches):
            idx = torch.nonzero(intersects[i]).flatten()
            if idx.numel() == 0:
                empty_boxes = torch.zeros(
                    (0, 2, 3), device=device, dtype=patch_min.dtype
                )
                names_list.append([])
                boxes_list.append(empty_boxes)
                continue

            boxes_i = obj_bboxes[idx] - patch_min[i][None, None, :]
            if obj_names is None:
                names_i = ["" for _ in range(idx.numel())]
            else:
                names_i = [obj_names[j] for j in idx.tolist()]
            names_list.append(names_i)
            boxes_list.append(boxes_i)

    result = copy.copy(scene_kwargs)  # shallow copy — scalars shared, not duplicated
    result["bbox_world"] = patch_bbox_world  # [N, 6]
    result["object_names"] = names_list  # [[str], ...] length N
    result["object_bboxes"] = boxes_list  # [Tensor [M_i, 2, 3], ...] length N

    return result


def split_fvdb(
    vdbtensor: fvnn.VDBTensor,
    patch_shape: Tuple[int, int, int],
    patch_bbox_worlds: torch.Tensor,  # [N, 6]
    scene_bbox_world: torch.Tensor,  # [1, 6]
    structure: Optional[torch.Tensor] = None,
    cpu_offload: bool = False,
) -> Tuple[fvnn.VDBTensor, Optional[torch.Tensor]]:
    """Split a single-batch scene fvdb into a batched fvnn.VDBTensor of patches.

    Args:
        vdbtensor: Scene VDBTensor with batch_size=1.
        patch_shape: Patch shape in voxels (Px, Py, Pz).
        patch_bbox_worlds: Tensor of bounding boxes for each patch.
        scene_bbox_world: Scene bounding box tensor.
        structure: Optional dense structure tensor [1, 1, Sx, Sy, Sz].

    Returns:
        band: fvnn.VDBTensor with batch_size = num_patches
        structures: Tensor [N, 1, Px, Py, Pz] or None
    """
    assert len(vdbtensor.grid) == 1, "split only supports batch_size=1"
    assert patch_bbox_worlds.shape[-1] == 6, "Invalid patch_bbox_worlds shape"
    assert scene_bbox_world.shape == (1, 6), "Invalid scene bbox_world shape"

    device = vdbtensor.device
    voxel_size: float = vdbtensor.voxel_sizes[0][0].item()
    patch_shape_t = torch.tensor(patch_shape, device=device, dtype=torch.long)
    scene_min = scene_bbox_world[0, :3].to(device)

    # grid.clip only works on CPU, so move there for clipping
    sub_grid_cpu = vdbtensor.grid[0].to("cpu")
    sub_data_cpu = vdbtensor.data[0].to("cpu")

    num_patches = len(patch_bbox_worlds)
    all_ijks = []
    all_values = []
    structures_list: List[torch.Tensor] = []

    # Pad structure if any patch extends beyond its bounds (from factor-aligned
    # last start rounding UP).
    if structure is not None:
        Sx, Sy, Sz = structure.shape[2], structure.shape[3], structure.shape[4]
        max_end = [0, 0, 0]
        for i in range(num_patches):
            sv = torch.round(
                (patch_bbox_worlds[i, :3].to(device) - scene_min) / voxel_size
            ).long()
            for d in range(3):
                end_d = sv[d].item() + patch_shape[d]
                if end_d > max_end[d]:
                    max_end[d] = end_d
        pad_x = max(0, max_end[0] - Sx)
        pad_y = max(0, max_end[1] - Sy)
        pad_z = max(0, max_end[2] - Sz)
        if pad_x > 0 or pad_y > 0 or pad_z > 0:
            structure = F.pad(
                structure, (0, pad_z, 0, pad_y, 0, pad_x), value=Voxel.EMPTY
            )

    for i in range(num_patches):
        patch_bbox = patch_bbox_worlds[i]  # [6]
        start_world = patch_bbox[:3].to(device)
        start_voxels = torch.round((start_world - scene_min) / voxel_size).long()

        start_int = start_voxels.to(dtype=torch.int32)
        end_int = start_int + patch_shape_t.to(device, dtype=torch.int32) - 1

        clipped_data, clipped_grid = sub_grid_cpu.clip(
            sub_data_cpu, [start_int.cpu().tolist()], [end_int.cpu().tolist()]
        )

        store_device = "cpu" if cpu_offload else device
        if clipped_grid.total_voxels == 0:
            # fVDB MaxPool CUDA kernel crashes on batch elements with 0 voxels
            # (illegal memory access). Add a single dummy voxel at the origin
            # with zero features — the encoder's structure masking will discard
            # it since the structure for empty regions is Voxel.EMPTY.
            coords_p = torch.zeros((1, 3), device=store_device, dtype=torch.int32)
            data_p = torch.zeros(
                (1,) + vdbtensor.data.jdata.shape[1:],
                device=store_device,
                dtype=vdbtensor.data.jdata.dtype,
            )
        else:
            coords_p = (clipped_grid.ijk.jdata - start_int.cpu()).to(store_device)
            data_p = clipped_data.jdata.to(store_device)

        all_ijks.append(coords_p)
        all_values.append(data_p)

        if structure is not None:
            sx, sy, sz = start_voxels.tolist()
            px, py, pz = patch_shape
            structures_list.append(
                structure[0:1, :, sx : sx + px, sy : sy + py, sz : sz + pz]
            )

    if cpu_offload:
        all_ijks = [t.to(device) for t in all_ijks]
        all_values = [t.to(device) for t in all_values]
    ijks = fvdb.JaggedTensor(all_ijks)
    values = fvdb.JaggedTensor(all_values)
    grid = fvdb.gridbatch_from_ijk(
        ijks,
        voxel_sizes=[voxel_size] * 3,
        origins=[voxel_size / 2.0] * 3,
    )
    inv = grid.ijk_to_inv_index(ijks)
    values = values[inv]

    batched_band = fvnn.VDBTensor(grid, values)
    batched_structure = torch.cat(structures_list, dim=0) if structures_list else None

    return batched_band, batched_structure


def split_structure(
    structure: torch.Tensor,
    patch_shape: Tuple[int, int, int],
    patch_bbox_worlds: torch.Tensor,
    scene_bbox_world: torch.Tensor,
    voxel_size: float,
) -> torch.Tensor:
    """Split a dense structure volume into patches.

    Args:
        structure: Dense structure tensor [B, 1, Sx, Sy, Sz].
        patch_shape: Patch shape in voxels (Px, Py, Pz).
        patch_bbox_worlds: World-space bboxes for each patch [N, 6].
        scene_bbox_world: Scene bounding box [B, 6].
        voxel_size: Voxel size in meters.

    Returns:
        Tensor of shape [B*N, 1, Px, Py, Pz], ordered as
        [b0_p0, b0_p1, ..., b0_pN-1, b1_p0, ..., bB-1_pN-1].
    """
    B = structure.shape[0]
    N = patch_bbox_worlds.shape[0]
    px, py, pz = patch_shape
    Sx, Sy, Sz = structure.shape[2], structure.shape[3], structure.shape[4]
    assert (
        Sx >= px and Sy >= py and Sz >= pz
    ), f"Structure shape ({Sx}, {Sy}, {Sz}) must be >= patch shape ({px}, {py}, {pz})"

    results = []
    for b in range(B):
        scene_min = scene_bbox_world[b, :3]
        for i in range(N):
            patch_min = patch_bbox_worlds[i, :3]
            start_voxels = torch.round((patch_min - scene_min) / voxel_size).long()
            sx, sy, sz = start_voxels.tolist()
            results.append(
                structure[b : b + 1, :, sx : sx + px, sy : sy + py, sz : sz + pz]
            )

    return torch.cat(results, dim=0)


def merge_fvdb(
    patches: fvnn.VDBTensor,
    patch_bbox_worlds: torch.Tensor,  # [N, 6]
    scene_latent_shape: Tuple[int, int, int],
    cpu_offload: bool = False,
) -> fvnn.VDBTensor:
    """Merge a batched fvnn.VDBTensor of patches back into a single scene by averaging overlaps.

    Args:
        patches: Batched fvnn.VDBTensor with batch_size = num_patches.
        patches_bbox_world: Tensor of bounding boxes for each patch.
        scene_latent_shape: Full scene shape in latent space [Sx, Sy, Sz].

    Returns:
        fvnn.VDBTensor: Single-batch merged scene with averaged overlapping regions.
    """
    device = patches.device
    voxel_size: float = patches.voxel_sizes[0][0].item()
    num_patches = len(patches.grid)

    # Get scene origin from first patch - compute scene_min from patch bboxes
    scene_min = patch_bbox_worlds[:, :3].min(dim=0).values  # [3]

    shape_x, shape_y, shape_z = scene_latent_shape
    plane = shape_x * shape_y

    # When cpu_offload is enabled, accumulate and average on CPU to save GPU memory
    work_device = "cpu" if cpu_offload else device

    all_indices = []
    all_values = []

    for i in range(num_patches):
        patch_grid = patches.grid[i]
        patch_data = patches.data[i]

        if patch_grid.total_voxels == 0:
            continue

        coords = patch_grid.ijk.jdata.to(work_device)
        values = patch_data.jdata.to(work_device)

        start_world = patch_bbox_worlds[i, :3].to(work_device)
        start_voxels = torch.round(
            (start_world - scene_min.to(work_device)) / voxel_size
        ).long()

        global_ijk = coords + start_voxels[None, :]
        linear = (
            global_ijk[:, 0] + global_ijk[:, 1] * shape_x + global_ijk[:, 2] * plane
        )
        all_indices.append(linear)
        all_values.append(values)

    if len(all_indices) == 0:
        # Return empty grid
        empty_ijk = torch.zeros((0, 3), device=device, dtype=torch.int32)
        empty_vals = torch.zeros(
            (0, patches.data.jdata.shape[-1]),
            device=device,
            dtype=patches.data.jdata.dtype,
        )
        ijks = fvdb.JaggedTensor([empty_ijk])
        vals = fvdb.JaggedTensor([empty_vals])
        grid = fvdb.gridbatch_from_ijk(
            ijks, voxel_sizes=[voxel_size] * 3, origins=[voxel_size / 2.0] * 3
        )
        return fvnn.VDBTensor(grid, vals)

    indices = torch.cat(all_indices, dim=0)
    values = torch.cat(all_values, dim=0)
    del all_indices, all_values
    unique_idx, inverse = torch.unique(indices, return_inverse=True)
    del indices
    num_unique = unique_idx.numel()
    feat_dim = values.shape[1] if values.ndim > 1 else 1

    sum_values = torch.zeros(
        (num_unique, feat_dim), device=work_device, dtype=values.dtype
    )
    count = torch.zeros((num_unique, 1), device=work_device, dtype=values.dtype)
    sum_values.index_add_(0, inverse, values if values.ndim > 1 else values[:, None])
    ones = torch.ones((values.shape[0], 1), device=work_device, dtype=values.dtype)
    count.index_add_(0, inverse, ones)
    del inverse, values, ones
    mean_values = sum_values / count.clamp_min(1)
    del sum_values, count
    if feat_dim == 1 and mean_values.ndim > 1:
        mean_values = mean_values.squeeze(1)

    x = unique_idx % shape_x
    y = (unique_idx // shape_x) % shape_y
    z = unique_idx // plane
    coords = torch.stack([x, y, z], dim=1).to(torch.int32).to(device)
    del unique_idx
    mean_values = mean_values.to(device)
    ijks = fvdb.JaggedTensor([coords])
    vals = fvdb.JaggedTensor([mean_values])
    grid = fvdb.gridbatch_from_ijk(
        ijks,
        voxel_sizes=[voxel_size] * 3,
        origins=[voxel_size / 2.0] * 3,
    )
    inv = grid.ijk_to_inv_index(ijks)
    vals = vals[inv]
    return fvnn.VDBTensor(grid, vals)


def unbind_kwargs(
    kwargs: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Split a single scene patch into a list of smaller Patch instances.

    Delegates to :func:`split_kwargs` and wraps each result in a Patch.

    Args:
        patch_shape: Patch shape in voxels (Px, Py, Pz).
        overlap: Overlap ratio (0-1) for each dimension.
        factor: VAE downsampling factor.

    Returns:
        List of Patch objects with batch_size=1.
    """
    N = len(kwargs["bbox_world"])
    kwargs_list = []
    for i in range(N):
        patch_kw = {k: v for k, v in kwargs.items()}
        patch_kw["bbox_world"] = kwargs["bbox_world"][i].unsqueeze(0)  # [1, 6]
        patch_kw["object_names"] = [kwargs["object_names"][i]]
        patch_kw["object_bboxes"] = [kwargs["object_bboxes"][i]]
        kwargs_list.append(patch_kw)
    return kwargs_list

class Patch:
    """Base class for spatial scene patches with object layout information.

    Represents a spatial region of a scene with associated bounding boxes, object names,
    and rendering parameters. Provides utilities for exporting, splitting, and visualizing
    scene layout.

    Attributes:
        kwargs: Dictionary containing patch parameters (bbox, voxel_size, objects, etc.).
        palette: Color palette for visualizing object categories.
    """

    def __init__(self, kwargs: Dict[str, Any]):
        """Initialize a Patch with configuration parameters.

        Args:
            kwargs: Dictionary containing:
                - bbox_world: World-space bounding box tensor [B, 6]
                - voxel_size: Voxel size in meters
                - object_names: List of object name lists
                - object_bboxes: List of object bbox tensors
                - num_categories: Total number of semantic categories
                - Additional metadata (known_ratio, truncation, etc.)
        """
        self.kwargs = kwargs
        self.palette, self._name_to_id = load_semantic_palette()

    @property
    def voxel_size(self) -> float:
        """Voxel size in meters."""
        return self.kwargs["voxel_size"]

    @property
    def ceiling_clip(self) -> float:
        """Height in meters at which to clip ceilings for rendering."""
        return self.kwargs["ceiling_clip"]

    @property
    def known_ratio(self) -> float:
        """Ratio threshold for determining known vs unknown voxels."""
        return self.kwargs["known_ratio"]

    @property
    def truncation(self) -> float:
        """TSDF truncation distance in meters."""
        return self.kwargs["truncation"]

    @property
    def bbox_world(self) -> torch.Tensor:
        """World-space bounding box tensor of shape [B, 6]."""
        return self.kwargs["bbox_world"]

    @property
    def object_names(self) -> List[List[str]]:
        """List of object name lists, one per batch element."""
        return self.kwargs["object_names"]

    @property
    def object_bboxes(self) -> List[torch.Tensor]:
        """List of object bbox tensors [M, 2, 3], one per batch element."""
        return self.kwargs["object_bboxes"]

    @property
    def object_categories(self) -> List[np.ndarray]:
        """Map object names to global category IDs from semantic_classes.csv.

        Returns:
            List of integer arrays mapping objects to category indices.
        """
        if not hasattr(self, "_category_cache"):
            self._category_cache = [
                np.array(
                    [self._name_to_id.get(name, 0) for name in names], dtype=np.int64
                )
                for names in self.object_names
            ]
        return self._category_cache

    @property
    def num_categories(self) -> int:
        """Total number of semantic categories."""
        return self.kwargs["num_categories"]

    @property
    def device(self) -> torch.device:
        """PyTorch device of the patch tensors."""
        return self.bbox_world.device

    @property
    def batch_size(self) -> int:
        """Number of patches in the batch."""
        return len(self.bbox_world)

    @property
    def patch_size(self) -> torch.Tensor:
        """Spatial size of patches in meters, shape [B, 3]."""
        return self.bbox_world[:, 3:] - self.bbox_world[:, :3]

    @property
    def patch_shape(self) -> List[int]:
        """Spatial dimensions of patches in voxels [D, H, W]."""
        if not hasattr(self, "_patch_shape"):
            self._patch_shape = get_patch_shape(self.bbox_world, self.voxel_size)
        return self._patch_shape

    def export_objects_as_mesh(
        self,
        out_dir: Optional[str] = None,
        scene_names: Optional[List[str]] = None,
        style: str = "wireframe",
        postfix: str = "",
    ) -> List[trimesh.Trimesh]:
        """Export object bounding boxes as colored mesh visualizations.

        Converts object bounding boxes to mesh representations with category-based
        coloring. Optionally saves meshes to disk.

        Args:
            out_dir: Optional directory to save mesh files. If None, meshes are
                returned but not saved.
            scene_names: List of scene names for organizing output files.
            postfix: String suffix to append to output filenames.

        Returns:
            List of trimesh.Trimesh objects, one per batch element, containing
            colored bounding box visualizations.

        Notes:
            Output filenames follow format: {bbox_min}_bbox_{postfix}.ply
        """
        meshes = []
        for idx in range(self.batch_size):
            bboxes = self.object_bboxes[idx]
            categories = self.object_categories[idx]

            if len(bboxes) > 0:
                bboxes = bboxes + self.bbox_world[idx][None, None, :3]  # [M, 2, 3]
                mesh = tc.bboxes2mesh(
                    bboxes, categories=categories, style=style, palette=self.palette
                )

            if out_dir is not None:
                bbox_str = bbox2str(self.bbox_world[idx])
                out_dir_i = os.path.join(out_dir, f"{scene_names[idx]}_{bbox_str}")
                os.makedirs(out_dir_i, exist_ok=True)
                mesh.export(os.path.join(out_dir_i, f"bbox_{postfix}.ply"))

            meshes.append(mesh)

        return meshes

    # Structural categories excluded from layout visualization by default
    STRUCTURAL_CATEGORIES = frozenset(
        {
            "void",
            "wall",
            "ceiling",
            "floor",
            "column",
            "beam",
            "baseboard",
            "lightband",
            "pipe",
            "stair",
        }
    )

    def export_objects_for_blender(
        self,
        alpha: float = 0.5,
        exclude_structural: bool = True,
    ) -> Tuple[List[List[trimesh.Trimesh]], List[List[Dict[str, Any]]]]:
        """Export per-object solid box meshes with CLIP-based colors for Blender rendering.

        Returns individual meshes and style dicts (one per object) suitable for
        mesh2images(..., mesh_styles=styles, backend="blender") which renders each
        object as a separate semi-transparent solid volume.

        Args:
            alpha: Opacity for the transparent material (0=invisible, 1=opaque).
            exclude_structural: If True, skip structural categories (wall, ceiling, etc.).

        Returns:
            Tuple of (meshes_per_batch, styles_per_batch) where each is
            List[List[...]] indexed by [batch][object].
        """
        all_meshes = []
        all_styles = []
        for idx in range(self.batch_size):
            bboxes = self.object_bboxes[idx]
            categories = self.object_categories[idx]
            names = self.object_names[idx]
            meshes_i = []
            styles_i = []

            if len(bboxes) > 0:
                bboxes_world = bboxes + self.bbox_world[idx][None, None, :3]
                for j, (bbox, cat_id, name) in enumerate(
                    zip(bboxes_world, categories, names)
                ):
                    if exclude_structural and name in self.STRUCTURAL_CATEGORIES:
                        continue
                    bbox_np = (
                        bbox.cpu().numpy() if isinstance(bbox, torch.Tensor) else bbox
                    )
                    size = bbox_np[1] - bbox_np[0]
                    center = (bbox_np[1] + bbox_np[0]) / 2
                    if size.min() > 0.0:
                        box = trimesh.creation.box(extents=size)
                        box.apply_translation(center)
                        meshes_i.append(box)

                        # Get color from palette or CLIP for unknown names
                        if cat_id < len(self.palette):
                            rgba = self.palette[cat_id]
                        else:
                            rgba = get_clip_color(name)
                        color = (rgba[0] / 255.0, rgba[1] / 255.0, rgba[2] / 255.0, 1.0)
                        styles_i.append(
                            {"color": color, "alpha": alpha, "roughness": 0.3}
                        )

            all_meshes.append(meshes_i)
            all_styles.append(styles_i)

        return all_meshes, all_styles

    def export_objects_as_json(
        self, out_dir: str, scene_names: List[str], postfix: str = ""
    ) -> None:
        """Export patch metadata and object information to JSON files.

        Saves patch parameters and object data in JSON format for later loading
        or analysis.

        Args:
            out_dir: Directory to save JSON files.
            scene_names: List of scene names for organizing output files.
            postfix: String suffix to append to output filenames.

        Notes:
            Output filenames follow format: {bbox_min}_meta_{postfix}.json
            JSON contains: bbox_world, voxel_size, object names/bboxes, and
            rendering parameters.
        """
        for idx in range(len(scene_names)):
            bbox_str = bbox2str(self.bbox_world[idx])
            out_dir_i = os.path.join(out_dir, f"{scene_names[idx]}_{bbox_str}")
            os.makedirs(out_dir_i, exist_ok=True)

            object_data = {
                "bbox_world": self.bbox_world[idx].cpu().numpy().tolist(),
                "voxel_size": self.voxel_size,
                "ceiling_clip": self.ceiling_clip,
                "known_ratio": self.known_ratio,
                "truncation": self.truncation,
                "object_names": self.object_names[idx],
                "object_bboxes": self.object_bboxes[idx].cpu().numpy().tolist(),
            }

            json_file = os.path.join(out_dir_i, f"meta_{postfix}.json")
            with open(json_file, "w") as f:
                json.dump(object_data, f, indent=2)

    def plot_bounding_boxes(
        self,
        num_views: int = 1,
        resolution: int = 150,
        **unused_kwargs,
    ) -> np.ndarray:
        """Plot 2D projections of object bounding boxes from multiple viewpoints.

        Renders top-down or angled views of the scene layout showing colored
        bounding boxes for all objects.

        Args:
            num_views: Number of views at different rotation angles around Z-axis.
            resolution: Image resolution in pixels (square images).
            **unused_kwargs: Additional kwargs for interface compatibility (ignored).

        Returns:
            NumPy array of shape [B, V, 3, H, W] containing rendered images where:
            - B is batch size
            - V is number of views
            - 3 is RGB channels
            - H, W are image height/width (both equal to resolution)
        """
        kwargs = {
            "object_bboxes": [x.cpu().numpy() for x in self.object_bboxes],
            "object_categories": self.object_categories,
            "object_names": self.object_names,
            "patch_size": self.patch_size.cpu().numpy(),
            "palette": self.palette,
            "resolution": resolution,
        }
        rotations = np.linspace(0, 360, num_views + 1)[:-1]
        images = []
        for rotation in rotations:
            kwargs["rotation"] = rotation
            images.append(plot_object_bboxes(**kwargs))
        images = torch.stack(images, dim=1)  # [B, V, 3, H, W]
        return images.cpu().numpy()

    def render_field(
        self,
        num_views: int = 1,
        resolution: int = 150,
        theta: float = 60.0,
        light_intensity: float = 5.0,
        return_meshes: bool = False,
        backend: str = "pyrender",
    ) -> np.ndarray:
        """Render photorealistic views of the scene using mesh representations.

        Exports the field as a mesh and renders it from multiple camera viewpoints
        using either PyRender or Blender.

        Args:
            num_views: Number of views at evenly spaced rotation angles.
            resolution: Image resolution in pixels (square images).
            theta: Camera elevation angle in degrees (0° = top-down, 90° = horizontal).
            light_intensity: Lighting strength for rendering.
            return_meshes: If True, also return the mesh objects.
            backend: Rendering backend, either "pyrender" (fast) or "blender" (high quality).

        Returns:
            NumPy array of shape [B, V, 3, H, W] containing rendered RGB images.
        """
        meshes = self.export_mesh()
        images = tc.mesh2images(
            meshes,
            num_views=num_views,
            resolution=resolution,
            bbox_world=self.bbox_world.cpu().numpy(),
            ceiling_clip=self.ceiling_clip,
            theta=theta,
            light_intensity=light_intensity,
            backend=backend,
        )

        if return_meshes:
            return images, meshes

        return images


class MeshPatch(Patch):
    def __init__(
        self,
        vertices: List[torch.Tensor],
        faces: List[torch.Tensor],
        kwargs: Dict[str, Any],
    ):
        super().__init__(kwargs)
        self.vertices = vertices
        self.faces = faces

    def export_mesh(
        self,
        out_dir: Optional[str] = None,
        scene_names: Optional[List[str]] = None,
        postfix: str = "",
    ) -> List[trimesh.Trimesh]:
        meshes = []
        for idx, (vertices, faces) in enumerate(zip(self.vertices, self.faces)):
            vertices = (
                vertices + self.bbox_world[idx][:3]
            )  # Shift vertices to world coordinates
            mesh = trimesh.Trimesh(
                vertices=vertices.cpu().numpy(), faces=faces.cpu().numpy()
            )
            meshes.append(mesh)

            if out_dir is not None and len(vertices) > 0 and len(faces) > 0:
                bbox_str = "_".join(
                    [f"{x:.01f}" for x in self.bbox_world[idx, :3].cpu().numpy()]
                )
                out_dir_i = os.path.join(out_dir, f"{scene_names[idx]}_{bbox_str}")
                os.makedirs(out_dir_i, exist_ok=True)
                mesh.export(os.path.join(out_dir_i, f"{postfix}.ply"))

        return meshes

    def surface_interc_grid(self) -> fvdb.GridBatch:
        vertices = fvdb.JaggedTensor(self.vertices)
        faces = fvdb.JaggedTensor(self.faces)
        grid = fvdb.set_from_mesh(
            vertices, faces, voxel_size=self.voxel_size, origin=self.voxel_size / 2.0
        )
        return grid

    def split(self, *args, **kwargs) -> List["MeshPatch"]:
        raise NotImplementedError("MeshPatch splitting not implemented yet")


class TSDFPatch(Patch):
    def __init__(
        self,
        band: fvnn.VDBTensor,
        structure: Optional[torch.Tensor] = None,
        kwargs: Dict[str, Any] = None,
    ):
        super().__init__(kwargs)
        self.band = band

        if structure is None:
            self.structure = torch.full(
                (len(band.grid), 1, *self.patch_shape),
                Voxel.EMPTY,
                device=band.device,
                dtype=torch.int8,
            )
        else:
            self.structure = structure

        assert (
            abs(self.voxel_size - band.voxel_sizes[0][0].item()) < 1e-6
        ), "Voxel size mismatch!"

    @property
    def known_mask(self) -> torch.Tensor:
        return self.structure != Voxel.UNKNOWN

    @property
    def band_mask(self) -> torch.Tensor:
        return self.structure == Voxel.BAND

    @property
    def surface_ratio(self) -> torch.Tensor:
        return self.band.grid.num_voxels / torch.tensor(self.patch_shape)

    @property
    def known_region_ratio(self) -> torch.Tensor:
        assert self.structure is not None
        return (self.structure != Voxel.UNKNOWN).float().mean(dim=(1, 2, 3, 4))

    @property
    def band_vdb(self) -> fvnn.VDBTensor:
        return self.band.clone()

    @torch.no_grad()
    def to_dense(self) -> torch.Tensor:
        volume = self.truncation * torch.ones_like(self.structure).float()
        volume[self.structure == Voxel.UNKNOWN] = -self.truncation
        volume[
            self.band.grid.jidx,
            :,
            self.band.grid.ijk.jdata[:, 0],
            self.band.grid.ijk.jdata[:, 1],
            self.band.grid.ijk.jdata[:, 2],
        ] = self.band.data.jdata

        return volume

    @torch.no_grad()
    def known_vdb(self) -> fvnn.VDBTensor:
        ijks = torch.nonzero(self.structure[:, 0] != Voxel.UNKNOWN)  # [N, 4]
        ijks = [ijks[ijks[:, 0] == b, 1:] for b in range(self.batch_size)]
        grid = fvdb.gridbatch_from_ijk(
            fvdb.JaggedTensor(ijks),
            voxel_sizes=self.band.grid.voxel_sizes,
            origins=self.band.grid.origins,
        )
        data = grid.fill_from_grid(self.band.data, self.band.grid, self.truncation)
        return fvnn.VDBTensor(grid, data)

    @torch.no_grad()
    def build_geometry_tree(self, tree_depth: int) -> Dict[int, fvnn.VDBTensor]:
        geometry_tree = {0: self.band.clone()}
        for depth in range(1, tree_depth):
            geometry_tree[depth] = fvnn.AvgPool(2)(geometry_tree[depth - 1])
        return geometry_tree

    @torch.no_grad()
    def build_structure_tree(self, tree_depth: int) -> Dict[int, torch.Tensor]:
        assert self.structure is not None, "Empty field is required for structure tree"

        structure_tree = {}
        structure = self.structure.to(torch.float16)
        for depth in range(tree_depth):
            structure_tree[depth] = structure
            structure = F.max_pool3d(structure, kernel_size=2, stride=2)

        return structure_tree

    @torch.no_grad()
    def build_field_tree(self, tree_depth: int) -> Dict[int, "TSDFPatch"]:
        geometry_tree = self.build_geometry_tree(tree_depth)
        structure_tree = self.build_structure_tree(tree_depth)

        field_tree = {}
        for feat_depth, geometry_vdb in geometry_tree.items():
            structure = structure_tree[feat_depth]
            field_params = copy.deepcopy(self.kwargs)
            field_params["voxel_size"] = geometry_vdb.voxel_sizes[0][0].item()
            field_tree[feat_depth] = TSDFPatch(
                geometry_vdb, structure, kwargs=field_params
            )

        return field_tree

    def export_mesh(
        self,
        out_dir: Optional[str] = None,
        scene_names: Optional[List[str]] = None,
        postfix: str = "",
    ) -> List[trimesh.Trimesh]:
        bbox_world = self.bbox_world.cpu().numpy()
        if len(self.band.grid.ijk.jdata) > 0:
            vertices, faces, _ = self.band.grid.marching_cubes(
                self.band.data, level=0.0
            )
            vertices = [
                vertices[i].jdata.cpu().numpy() + bbox_world[i, :3]
                for i in range(len(self.band.grid))
            ]
            faces = [f.jdata.cpu().numpy() for f in faces]
        else:
            vertices = [np.zeros((0, 3))] * len(self.band.grid)
            faces = [np.zeros((0, 3))] * len(self.band.grid)
            logger.warning("No vertices found in the grid")

        meshes = []
        for idx in range(len(self.band.grid)):
            mesh = trimesh.Trimesh(vertices[idx], faces[idx], process=False)
            meshes.append(mesh)

        if out_dir is not None:
            for idx, mesh in enumerate(meshes):
                bbox_str = bbox2str(self.bbox_world[idx])
                out_dir_i = os.path.join(out_dir, f"{scene_names[idx]}_{bbox_str}")
                os.makedirs(out_dir_i, exist_ok=True)
                mesh.export(os.path.join(out_dir_i, f"mesh_{postfix}.ply"))

        return meshes

    def extract_structure(self) -> List[trimesh.Trimesh]:
        volume = torch.zeros(
            (len(self.band.grid), 1, *self.patch_shape), device=self.band.device
        )
        indices = self.band.grid.ijk.jdata
        volume[self.band.grid.jidx, :, indices[:, 0], indices[:, 1], indices[:, 2]] = 1

        volume = volume.float().cpu().numpy()
        bbox_world = self.bbox_world.cpu().numpy()

        meshes = []
        for idx, volume_i in enumerate(volume):
            if volume_i.min() >= 1.0e-4 or volume_i.max() <= 1.0e-4:
                mesh = trimesh.Trimesh(np.zeros((0, 3)), np.zeros((0, 3)))
                logger.warning(f"No mesh found for {idx}")
            else:
                mesh = tc.volume2mesh(
                    volume_i[0],
                    isovalue=1.0e-4,
                    bbox_min=bbox_world[idx, :3],
                    voxel_size=self.voxel_size,
                )
            meshes.append(mesh)

        return meshes

    def export_scenes_as_npz(
        self, out_dir: str, scene_names: List[str], postfix: str = ""
    ):
        scene_data = []
        for idx in range(len(self.band.grid)):
            bbox_str = bbox2str(self.bbox_world[idx])
            out_dir_i = os.path.join(out_dir, f"{scene_names[idx]}_{bbox_str}")
            os.makedirs(out_dir_i, exist_ok=True)

            kwargs = {
                "ijks": self.band.grid[idx].ijk.jdata.cpu().numpy(),
                "values": self.band.data[idx].jdata.cpu().numpy(),
            }
            if self.structure is not None:
                structure = self.structure[idx, 0]  # [G1, G2, G3]
                kwargs |= {
                    "unknown": torch.nonzero(structure == Voxel.UNKNOWN).cpu().numpy(),
                    "empty": torch.nonzero(structure == Voxel.EMPTY).cpu().numpy(),
                }

            np.savez_compressed(
                os.path.join(out_dir_i, f"geometry_{postfix}.npz"), **kwargs
            )

        return scene_data

    def export_volumes(self, out_dir: str, scene_names: List[str], postfix: str = ""):
        volume = self.to_dense().cpu().numpy()  # [B, 1, G1, G2, G3]
        for idx in range(len(self.band.grid)):
            bbox_str = bbox2str(self.bbox_world[idx])
            out_dir_i = os.path.join(out_dir, f"{scene_names[idx]}_{bbox_str}")
            os.makedirs(out_dir_i, exist_ok=True)
            # [1, G1, G2, G3]
            np.save(os.path.join(out_dir_i, f"volume_{postfix}.npy"), volume[idx])

    def render_structure(
        self,
        num_views: int = 1,
        resolution: int = 150,
        theta: float = 60.0,
        light_intensity: float = 5.0,
        backend: str = "pyrender",
    ) -> np.ndarray:
        meshes = self.extract_structure()
        images = tc.mesh2images(
            meshes,
            num_views=num_views,
            resolution=resolution,
            bbox_world=self.bbox_world.cpu().numpy(),
            ceiling_clip=self.ceiling_clip,
            theta=theta,
            light_intensity=light_intensity,
            backend=backend,
        )
        return images
