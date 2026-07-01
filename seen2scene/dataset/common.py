from typing import *
import torch
import numpy as np
import trimesh
import json
import csv
import os
import glob

from seen2scene.tools.log_utils import get_logger
from seen2scene.tools.common_utils import list_to_element
from seen2scene.tools.vdb_utils import vdbtensor_from_coords
from seen2scene.models.fields import Patch, MeshPatch, TSDFPatch

logger = get_logger(file_name=__file__, debug="dataset")


def split_file_paths(
    num_samples: int, ratios: Tuple[float] = [0.8, 0.05, 0.15], split: str = "all"
) -> np.ndarray:
    """Split dataset indices into train/val/test splits based on specified ratios.

    Args:
        num_samples: Total number of samples in the dataset.
        ratios: Tuple of 3 floats specifying [train_ratio, val_ratio, test_ratio].
            Default [0.8, 0.05, 0.15] means 80% train, 5% val, 15% test.
        split: Which split to return. Options: "train", "val", "test", or "all".
            "all" returns all indices.

    Returns:
        NumPy array of indices for the specified split.

    Raises:
        ValueError: If split is not one of "train", "val", "test", or "all".

    Examples:
        >>> indices = split_file_paths(100, ratios=[0.8, 0.1, 0.1], split="train")
        >>> len(indices)
        80
        >>> indices = split_file_paths(100, split="val")
        >>> len(indices)
        5
    """
    indices = np.arange(num_samples)
    val_nums = [int(num_samples * x) for x in ratios]
    val_nums[-1] = num_samples - sum(val_nums[:-1])
    if split == "train":
        indices = indices[: val_nums[0]]
    elif split == "val":
        indices = indices[val_nums[0] : val_nums[0] + val_nums[1]]
    elif split == "test":
        indices = indices[val_nums[0] + val_nums[1] :]
    elif split == "all":
        pass
    else:
        raise ValueError

    return indices


def determine_patch_region(
    bbox: torch.Tensor, chunk_size: torch.Tensor
) -> torch.Tensor:
    """Determine valid region for patch sampling within a bounding box.

    Computes a valid sampling region that ensures patches of size chunk_size can be
    extracted. For axes where the bbox is larger than chunk_size, leaves margin on
    both sides. For smaller axes, centers the region.

    Per-axis logic:
    - If bbox size > chunk_size: Valid region is [bbox_min + chunk_size/2, bbox_max - chunk_size/2]
    - If bbox size <= chunk_size: Valid region collapses to center point [chunk_size/2, chunk_size/2]

    Args:
        bbox: Bounding box tensor of shape [6] as [xmin, ymin, zmin, xmax, ymax, zmax].
        chunk_size: Patch dimensions as tensor of shape [3] containing [depth, height, width].

    Returns:
        Tensor of shape [2, 3] containing [min_coords, max_coords] of the valid
        sampling region.

    Examples:
        >>> bbox = torch.tensor([0., 0., 0., 10., 10., 10.])
        >>> chunk = torch.tensor([4., 4., 4.])
        >>> region = determine_patch_region(bbox, chunk)
        >>> region.shape
        torch.Size([2, 3])
    """
    # Calculate the size of the bounding box for each axis
    size = bbox[3:] - bbox[:3]
    new_min = bbox[:3] + chunk_size / 2
    new_max = torch.where(size > chunk_size, bbox[3:] - chunk_size / 2, new_min)
    return torch.stack([new_min, new_max])  # [2, 3]


def compute_intersected_objects(
    object_bboxes: torch.Tensor, chunk_bbox: torch.Tensor
) -> List[int]:
    """Compute which objects intersect with a given spatial chunk.

    Uses axis-aligned bounding box (AABB) intersection testing. Two bboxes intersect
    if and only if they overlap on all three axes simultaneously.

    Args:
        object_bboxes: Tensor of shape [M, 2, 3] where M is the number of objects.
            Each object bbox is [[xmin, ymin, zmin], [xmax, ymax, zmax]].
        chunk_bbox: Tensor of shape [6] representing the chunk bbox as
            [xmin, ymin, zmin, xmax, ymax, zmax].

    Returns:
        List of integer indices for objects that intersect with the chunk.
        Empty list if no objects intersect.

    Notes:
        Two axis-aligned bounding boxes intersect if for each axis, the ranges overlap:
        (obj_min <= chunk_max) AND (obj_max >= chunk_min)

    Examples:
        >>> obj_bboxes = torch.tensor([[[0., 0., 0.], [2., 2., 2.]],
        ...                            [[5., 5., 5.], [7., 7., 7.]]])
        >>> chunk = torch.tensor([1., 1., 1., 3., 3., 3.])
        >>> compute_intersected_objects(obj_bboxes, chunk)
        [0]  # Only first object intersects
    """
    # Extract min and max coordinates for both object bboxes and chunk bbox
    obj_mins = object_bboxes[:, 0, :]  # [M, 3] - object min coordinates
    obj_maxs = object_bboxes[:, 1, :]  # [M, 3] - object max coordinates
    chunk_min = chunk_bbox[:3]  # [3] - chunk min coordinates
    chunk_max = chunk_bbox[3:]  # [3] - chunk max coordinates

    # Check for intersection: two bboxes intersect if they overlap on all axes
    # For each axis, check if the ranges overlap
    intersects_x = (obj_mins[:, 0] <= chunk_max[0]) & (obj_maxs[:, 0] >= chunk_min[0])
    intersects_y = (obj_mins[:, 1] <= chunk_max[1]) & (obj_maxs[:, 1] >= chunk_min[1])
    intersects_z = (obj_mins[:, 2] <= chunk_max[2]) & (obj_maxs[:, 2] >= chunk_min[2])

    # Object intersects with chunk if it intersects on all three axes
    intersects = intersects_x & intersects_y & intersects_z

    # Get indices of intersecting objects
    intersected_indices = torch.where(intersects)[0].tolist()

    return intersected_indices


def custom_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Custom collate function for batching heterogeneous data types.

    Collates a list of sample dictionaries into a batch dictionary. Stacks tensors
    and arrays, but keeps other types (strings, objects) as lists.

    Args:
        batch: List of dictionaries where each dict represents one sample with
            the same set of keys.

    Returns:
        Dictionary with same keys as input samples. Values are:
        - torch.Tensor: Stacked using torch.stack if all values are tensors
        - np.ndarray: Stacked using np.stack if all values are arrays
        - Other types: Kept as lists

    Examples:
        >>> batch = [
        ...     {"tensor": torch.randn(3), "name": "sample1"},
        ...     {"tensor": torch.randn(3), "name": "sample2"}
        ... ]
        >>> collated = custom_collate_fn(batch)
        >>> collated["tensor"].shape
        torch.Size([2, 3])
        >>> collated["name"]
        ['sample1', 'sample2']
    """
    collated_batch = {}  # Output dictionary
    # Iterate over keys in the first sample (assuming all samples have the same keys)
    for key in batch[0].keys():
        values = [sample[key] for sample in batch]  # Extract values for this key
        if isinstance(values[0], torch.Tensor):
            collated_batch[key] = torch.stack(values)
        elif isinstance(values[0], np.ndarray):
            collated_batch[key] = np.stack(values)
        else:
            collated_batch[key] = values

    return collated_batch


def batch_to_LayoutPatch(
    bbox_world: Optional[torch.Tensor] = None,
    object_names: Optional[List[str]] = None,
    object_bboxes: Optional[List[List[float]]] = None,
    **kwargs,
) -> Patch:
    """Construct a layout Patch from batched data.

    Creates a Patch object containing scene layout information (bounding boxes
    and object names) without geometric data.

    Args:
        bbox_world: List of world-space bounding box tensors, one per sample.
        object_names: List of object names for layout conditioning.
        object_bboxes: List of object bounding boxes for each sample.
        **kwargs: Additional metadata passed to Patch constructor. List values
            are reduced to their first element.

    Returns:
        Patch object containing stacked bounding boxes and layout metadata.

    Examples:
        >>> bboxes = [torch.tensor([0., 0., 0., 10., 10., 10.])]
        >>> names = [['chair', 'table']]
        >>> patch = batch_to_LayoutPatch(bbox_world=bboxes, object_names=names)
    """
    kwargs = {k: list_to_element(v) for k, v in kwargs.items()}
    kwargs |= {
        "bbox_world": torch.stack(bbox_world),
        "object_names": object_names,
        "object_bboxes": object_bboxes,
    }
    return Patch(kwargs=kwargs)


def batch_to_MeshField(
    verts: List[torch.Tensor],
    faces: List[torch.Tensor],
    bbox_world: Optional[torch.Tensor] = None,
    object_names: Optional[List[str]] = None,
    object_bboxes: Optional[List[List[float]]] = None,
    **kwargs,
) -> MeshPatch:
    """Construct a MeshPatch from batched mesh data.

    Creates a MeshPatch object containing batched mesh geometry with associated
    spatial and semantic metadata.

    Args:
        verts: List of vertex tensors [N_i, 3] (float32), one per sample.
        faces: List of face tensors [F_i, 3] (int64), one per sample.
        bbox_world: List of world-space bounding box tensors.
        object_names: List of object names for each sample.
        object_bboxes: List of object bounding boxes for each sample.
        **kwargs: Additional metadata passed to MeshPatch constructor. List values
            are reduced to their first element.

    Returns:
        MeshPatch object containing batched mesh geometry with stacked metadata.
    """
    kwargs = {k: list_to_element(v) for k, v in kwargs.items()}
    kwargs |= {
        "bbox_world": torch.stack(bbox_world),
        "object_names": object_names,
        "object_bboxes": object_bboxes,
    }
    return MeshPatch(verts, faces, kwargs=kwargs)


def batch_to_fVDBField(
    band_coords: List[torch.Tensor],
    band_values: List[torch.Tensor],
    structure: Optional[List[torch.Tensor]] = None,
    voxel_size: Optional[List[float]] = None,
    bbox_world: Optional[List[torch.Tensor]] = None,
    object_names: Optional[List[str]] = None,
    object_bboxes: Optional[List[List[float]]] = None,
    **kwargs,
) -> TSDFPatch:
    """Construct a TSDFPatch from batched sparse voxel data.

    Creates a TSDFPatch using fVDB sparse voxel representation. Converts lists
    of coordinates and TSDF values into a batched VDBTensor.

    Args:
        band_coords: List of coordinate tensors for occupied voxels in the truncation band.
        band_values: List of TSDF value tensors corresponding to band_coords.
        structure: Optional list of structure masks indicating structural elements
            (walls, floors, ceilings).
        voxel_size: List containing voxel size in meters (same for all samples).
        bbox_world: List of world-space bounding box tensors.
        object_names: List of object names for each sample.
        object_bboxes: List of object bounding boxes for each sample.
        **kwargs: Additional metadata passed to TSDFPatch constructor. List values
            are reduced to their first element.

    Returns:
        TSDFPatch object containing fVDB sparse TSDF representation with metadata.

    Notes:
        The VDB origin is offset by voxel_size/2 to center voxels on integer coordinates.

    Examples:
        >>> coords = [torch.tensor([[0, 0, 0], [1, 1, 1]])]
        >>> values = [torch.tensor([[0.5], [-0.3]])]
        >>> voxel_sizes = [0.01]
        >>> patch = batch_to_fVDBField(coords, values, voxel_size=voxel_sizes)
    """
    voxel_size = voxel_size[0]
    band_vdb = vdbtensor_from_coords(
        band_coords,
        band_values,
        voxel_sizes=[voxel_size] * 3,
        origins=[voxel_size / 2.0] * 3,
    )
    if structure is not None:
        structure = torch.stack(structure)

    kwargs = {k: list_to_element(v) for k, v in kwargs.items()}
    kwargs |= {
        "voxel_size": band_vdb.grid.voxel_sizes[0][0].item(),
        "bbox_world": torch.stack(bbox_world),
        "object_names": object_names,
        "object_bboxes": object_bboxes,
    }
    return TSDFPatch(band=band_vdb, structure=structure, kwargs=kwargs)


def batch_to_Field(batch: Dict[str, Any]) -> Dict[str, Any]:
    """Convert batched field data dictionaries to Field objects.

    Dispatches to appropriate field constructor (MeshPatch, TSDFPatch, or Patch)
    based on the "field" type specified in the data. Processes all keys in the
    batch that contain field data.

    Args:
        batch: Dictionary containing batched data. Must include "field_params" key
            with list of parameter dictionaries. Other keys may contain field-specific
            data with "field" type indicator.

    Returns:
        Modified batch dictionary with field data converted to Field objects
        (MeshPatch, TSDFPatch, or Patch).

    Raises:
        ValueError: If field_name is not one of "mesh", "layout", or "tsdf".

    Notes:
        The "field_params" key is consumed and removed from the batch.

    Examples:
        >>> batch = {
        ...     "field_params": [{"voxel_size": 0.01, "bbox_world": bbox}],
        ...     "tsdf_gt": [{"field": "tsdf", "band_coords": coords, "band_values": vals}]
        ... }
        >>> batch = batch_to_Field(batch)
        >>> isinstance(batch["tsdf_gt"], TSDFPatch)
        True
    """
    field_params = batch.pop("field_params")  # List[Dict[str, Any]]
    field_params = {
        key: [d[key] for d in field_params] for key in field_params[0].keys()
    }

    for key, value in batch.items():
        if isinstance(value[0], dict) and "field" in value[0].keys():
            field_data = {k: [d[k] for d in value] for k in value[0].keys()}
            field_name = field_data.pop("field")[0]

            if field_name == "mesh":
                field = batch_to_MeshField(**field_params, **field_data)
            elif field_name == "layout":
                field = batch_to_LayoutPatch(**field_params, **field_data)
            elif field_name == "tsdf":
                field = batch_to_fVDBField(**field_params, **field_data)
            else:
                raise ValueError(f"Unknown field type: {field_name}")

            batch[key] = field

    return batch


def parse_bool(s: str) -> bool:
    """Parse a string representation of a boolean value.

    Case-insensitive comparison. Strips whitespace before checking.

    Args:
        s: String to parse (e.g., "true", "True", "TRUE", "false").

    Returns:
        True if the lowercase stripped string equals "true", False otherwise.

    Examples:
        >>> parse_bool("true")
        True
        >>> parse_bool("  True  ")
        True
        >>> parse_bool("false")
        False
        >>> parse_bool("anything_else")
        False
    """
    return s.strip().lower() == "true"


def load_scene_meta(
    meta_path: str,
    bottom_padding: float,
    voxel_size: float,
    chunk_size: torch.Tensor,
    semantic_classes: Optional[Dict[int, str]] = None,
    mapping_dict: Optional[Dict[str, int]] = None,
    clean_dict: Optional[Dict[str, str]] = None,
    category_exclude: Optional[Tuple[str, ...]] = None,
    category_include: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Load and process scene metadata from a JSON file.

    Reads scene metadata including bounding boxes and object information, applies
    semantic mapping, filters objects by category, and adjusts spatial bounds.

    Args:
        meta_path: Path to the scene's meta.json file.
        bottom_padding: Additional padding in voxels to add below the scene.
        voxel_size: Size of voxels in meters.
        chunk_size: Desired chunk dimensions as tensor of shape [3].
        semantic_classes: Dictionary mapping category IDs to category names.
        mapping_dict: Dictionary mapping raw object IDs/names to semantic category IDs.
        category_exclude: Optional tuple of category names to exclude from the scene.
        category_include: Optional specific category name that must be present. If set,
            scenes without this category are filtered out.

    Returns:
        Dictionary containing scene metadata:
        - "scene_name": Scene identifier
        - "scene_bounds": Adjusted scene bounding box tensor [6]
        - "object_names": List of (mapped) object names
        - "object_names_raw": List of original object names
        - "object_bboxes": Tensor of object bounding boxes [N, 2, 3]
        Returns None if no valid objects remain after filtering.

    Notes:
        - Scene bounds are adjusted: bottom is lowered by bottom_padding, top is
          clamped to bottom + chunk_size[2]
        - Objects above the adjusted ceiling are filtered out
        - All object names are converted to lowercase

    Examples:
        >>> meta = load_scene_meta(
        ...     "path/to/meta.json",
        ...     bottom_padding=3.0,
        ...     voxel_size=0.01,
        ...     chunk_size=torch.tensor([256, 256, 256]),
        ...     mapping_dict={},
        ...     semantic_classes={}
        ... )
    """
    scene_name = os.path.basename(os.path.dirname(meta_path))
    with open(meta_path, "r") as meta_file:
        meta_data = json.load(meta_file)

    def _flatten_box(box):
        """Flatten nested scene_box (e.g. [0, 0, 0, [4, 4, 3]] or [[0,0,0],[4,4,3]])."""
        flat = []
        for item in box:
            if isinstance(item, list):
                flat.extend(item)
            else:
                flat.append(item)
        return flat

    scene_bounds = torch.tensor(_flatten_box(meta_data["scene_box"]), dtype=torch.float32)
    scene_bounds[2] -= bottom_padding * voxel_size
    scene_bounds[5] = scene_bounds[2] + chunk_size[-1]

    # Clamp scene bounds so each dimension is at least 256 voxels
    # (prevents structure < patch_shape in split_structure)
    min_extent = 256 * voxel_size
    for dim in range(3):
        extent = scene_bounds[dim + 3] - scene_bounds[dim]
        if extent < min_extent:
            scene_bounds[dim + 3] = scene_bounds[dim] + min_extent

    if "object_names" not in meta_data:
        print(f"{meta_path} keys: {meta_data.keys()}!")

    names, names_raw, bboxes, volumes = [], [], [], []
    for idx in range(len(meta_data["object_names"])):
        object_bbox = torch.tensor(
            meta_data["object_bboxes"][idx], dtype=torch.float32
        )  # [2, 3]
        if clean_dict is not None:
            old_id = meta_data["object_categories"][idx]
            name_raw = clean_dict[str(old_id)]
        else:
            name_raw = meta_data["object_names"][idx]

        if semantic_classes is not None:
            category_id = mapping_dict.get(name_raw, 0)
            name = semantic_classes.get(category_id, "void")
        else:
            name = name_raw

        name = name.lower()  # Always use lowercase

        if category_exclude is not None and name in category_exclude:
            logger.debug(f"Excluding {name} from {scene_name}!")
            continue

        # Since scene bounds is clipped, filter out objects that are above the scene bounds
        if object_bbox[0][-1] > scene_bounds[-1]:
            logger.debug(
                f"{scene_name}: {name} at {object_bbox[0][-1]} is above ceiling {scene_bounds[-1]}!"
            )
            continue

        volume = torch.prod(object_bbox[1] - object_bbox[0]).item()

        names.append(name)
        bboxes.append(object_bbox)
        names_raw.append(name_raw)
        volumes.append(volume)

    if len(bboxes) == 0:
        logger.debug(f"No valid objects in {scene_name}!")
        return None

    if category_include is not None and category_include not in names:
        logger.debug(f"Excluding scene {scene_name} without {category_include}!")
        return None

    return {
        "scene_name": scene_name,
        "scene_bounds": scene_bounds,
        "object_names": names,
        "object_names_raw": names_raw,
        "object_bboxes": torch.stack(bboxes),
        "object_volumes": volumes,
    }


def load_and_split_data(
    data_name: str,
    root_dir: str,
    voxel_size: float,
    data_keys: Tuple[str, ...],
    bottom_padding: float,
    chunk_size: Tuple[float, float, float],
    overfit_scene: Optional[Union[str, Tuple[str, ...]]] = None,
    split: str = "all",
    fixed_len: int = -1,
    fusion_dir: Optional[Dict[str, str]] = None,
    mapping_path: Optional[str] = None,
    scene_exclude: Optional[Tuple[str, ...]] = None,
    category_exclude: Optional[Tuple[str, ...]] = None,
    category_include: Optional[Tuple[str, ...]] = None,
    use_semantic_mapping: bool = True,
    keep_region: Optional[str] = None,
) -> Tuple[List[Dict[str, str]], List[Dict[str, Any]]]:
    """Load dataset file paths and metadata with train/val/test splitting.

    Discovers scenes in root_dir, loads semantic mappings, validates data file
    existence, and splits into train/val/test sets. Supports overfitting to a
    single scene and various filtering options.

    Args:
        root_dir: Root directory containing scene subdirectories.
        voxel_size: Voxel size in meters for loading appropriate fusion data.
        data_keys: Tuple of data types to load (e.g., "tsdf_gt", "mesh_gt").
        bottom_padding: Additional padding in voxels below scenes.
        chunk_size: Chunk dimensions as tensor [3] for spatial bounds adjustment.
        overfit_scene: Optional scene name to use exclusively (for debugging/overfitting).
        split: Data split to return: "train", "val", "test", or "all".
        fixed_len: Maximum number of scenes to include. -1 means no limit.
        fusion_dir: Dictionary mapping voxel size strings to fusion data directories.
        mapping_path: Path to CSV file mapping dataset-specific IDs to semantic IDs.
        scene_exclude: Optional tuple of scene names to exclude.
        category_exclude: Optional tuple of object categories to exclude.
        category_include: Optional category that must be present in scenes.
        use_semantic_mapping: If True, map object names to semantic categories.

    Returns:
        Tuple of (file_paths, scene_metas):
        - file_paths: List of dicts containing paths to data files for each scene.
          Keys include "meta_path" and paths for each data_key (e.g., "tsdf_gt_path").
        - scene_metas: List of dicts containing scene metadata (from load_scene_meta).

    Notes:
        - Loads semantic class definitions from ./assets/semantic_classes.csv
        - Validates that all required data files exist before including a scene
        - Scenes without valid objects (after filtering) are excluded
        - Split ratios are fixed at [0.8, 0.05, 0.15] for train/val/test

    Examples:
        >>> paths, metas = load_and_split_data(
        ...     root_dir="/path/to/scenes",
        ...     fusion_dir={"0.011": "/path/to/fusion"},
        ...     voxel_size=0.011,
        ...     data_keys=("tsdf_gt",),
        ...     mapping_path="./assets/dataset_mapping.csv",
        ...     bottom_padding=3.0,
        ...     chunk_size=torch.tensor([256, 256, 256]),
        ...     split="train"
        ... )
        >>> len(paths), len(metas)
        (800, 800)  # 80% of 1000 total scenes
    """
    # Classes mapping
    mapping_dict = None
    if mapping_path is not None:
        mapping_dict = {}
        with open(mapping_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                mapping_dict[row["name"]] = int(row["id"])

    # Shared semantic classes
    semantic_classes = None
    if use_semantic_mapping:
        assert (
            mapping_dict is not None
        ), "mapping_dict must be provided if use_semantic_mapping is True!"
        semantic_classes = {}
        from seen2scene import ASSETS_DIR

        with open(ASSETS_DIR / "semantic_classes.csv", "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                semantic_classes[int(row["id"])] = row["name"]

    # Classes clean (Only for 3D-Front)
    clean_dict = None
    if data_name == "3D-FRONT":
        clean_dict = {}
        with open(mapping_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                clean_dict[str(row["old_id"])] = row["name"]

    scene_pathes = sorted(glob.glob(os.path.join(root_dir, "*")))

    scene_names, file_paths, scene_metas = [], [], []
    for scene_dir in scene_pathes:
        scene_name = os.path.basename(scene_dir)
        if scene_exclude is not None and scene_name in scene_exclude:
            logger.debug(f"Excluding {scene_name} from {root_dir}!")
            continue

        file_path = {"meta_path": os.path.join(scene_dir, "meta.json")}
        exists = {"meta": os.path.exists(file_path["meta_path"])}

        if data_keys is not None:
            for data_key in data_keys:
                if data_key == "mesh_gt":
                    data_path = os.path.join(scene_dir, "scene.ply")
                    if os.path.exists(data_path):
                        file_path[f"{data_key}_path"] = data_path
                elif data_key.startswith("tsdf_"):
                    data_name = (
                        f"fusion{data_key[data_key.index('_'):]}_v_{voxel_size:.3f}"
                    )
                    data_path = os.path.join(
                        fusion_dir[f"{voxel_size:.3f}"], scene_name, f"{data_name}.vdb"
                    )
                    file_path[f"{data_key}_path"] = data_path
                    exists[data_key] = os.path.exists(data_path)

        if all(value for value in exists.values()):
            scene_meta = load_scene_meta(
                meta_path=file_path["meta_path"],
                bottom_padding=bottom_padding,
                voxel_size=voxel_size,
                chunk_size=chunk_size,
                semantic_classes=semantic_classes,
                mapping_dict=mapping_dict,
                clean_dict=clean_dict,
                category_exclude=category_exclude,
                category_include=category_include,
            )
            if scene_meta is None:
                continue

            scene_meta |= {"keep_region": keep_region}
            file_paths.append(file_path)
            scene_names.append(scene_name)
            scene_metas.append(scene_meta)

    logger.debug(f"Found {len(file_paths)} files in {root_dir}!")

    if overfit_scene is not None:
        if isinstance(overfit_scene, str):
            overfit_scene = (overfit_scene,)
        indices = [scene_names.index(s) for s in overfit_scene if s in scene_names]
        if indices:
            file_paths = [file_paths[idx] for idx in indices]
            scene_metas = [scene_metas[idx] for idx in indices]
        else:
            file_paths, scene_metas = [], []
    else:
        indices = split_file_paths(len(file_paths), split=split)
        file_paths = [file_paths[i] for i in indices]
        scene_metas = [scene_metas[i] for i in indices]

    if fixed_len > 0:
        file_paths = file_paths[:fixed_len]
        scene_metas = scene_metas[:fixed_len]

    return file_paths, scene_metas
