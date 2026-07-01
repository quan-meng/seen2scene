import csv
import os
import torch
import trimesh
import numpy as np
from typing import *

from seen2scene.configs.dataset import Voxel, DATA_DICT
from .common import (
    determine_patch_region,
    compute_intersected_objects,
    load_and_split_data,
)

from seen2scene.tools.log_utils import get_logger
from seen2scene.tools.common_utils import interleave_lists, clip_mesh
import seen2scene.tools.augment_utils as tau
from seen2scene.tools.vdb_utils import loadvdb, crop_vdb

logger = get_logger(file_name=__file__, debug="dataset")


class Augmentations:
    """Data augmentation handler for 3D volumes and bounding boxes.

    Randomly applies rotation (90° multiples around Z-axis) and flipping
    augmentations with consistent transformations across volumes and spatial
    coordinates.

    Attributes:
        flip_dim: Dimension along which to apply flipping (X or Y axis).
        rotate_time: Number of 90° rotations to apply (1, 2, or 3 for 90°, 180°, 270°).
        flip_flag: Whether to apply flipping augmentation.
        rotate_flag: Whether to apply rotation augmentation.
    """

    def __init__(
        self,
        rng: np.random.Generator,
        flip_dims: List[int] = [-2, -3],
        rotate_times: List[int] = [1, 2, 3],
    ):
        """Initialize augmentation parameters with random choices.

        Args:
            rng: NumPy random number generator for reproducible augmentation.
            flip_dims: List of dimensions available for flipping. Default [-2, -3]
                corresponds to X or Y axes in spatial dimensions.
            rotate_times: List of possible 90° rotation counts. [1, 2, 3] means
                90°, 180°, or 270° rotations.

        Notes:
            Each augmentation (flip/rotate) is applied with 50% probability.
        """
        self.flip_dim = rng.choice(flip_dims)  # flip along x or y axis
        self.rotate_time = rng.choice(rotate_times)

        self.flip_flag = rng.random() < 0.5
        self.rotate_flag = rng.random() < 0.5

    def update_volume(self, volume: torch.Tensor) -> torch.Tensor:
        """Apply consistent augmentations to a 3D volume tensor.

        Applies rotation and/or flipping based on the randomly initialized
        augmentation parameters. Transformations are applied in order: rotation
        first, then flipping.

        Args:
            volume: Input volume tensor of shape [..., H, W, D] where last 3
                dimensions are spatial. Can have arbitrary leading batch/channel
                dimensions.

        Returns:
            Augmented volume tensor with same shape as input.

        Examples:
            >>> rng = np.random.default_rng(seed=42)
            >>> aug = Augmentations(rng)
            >>> volume = torch.randn(1, 1, 64, 64, 64)
            >>> augmented = aug.update_volume(volume)
            >>> augmented.shape
            torch.Size([1, 1, 64, 64, 64])
        """
        if self.rotate_flag:
            volume = tau.volume_rotz90(volume, self.rotate_time)
        if self.flip_flag:
            volume = tau.volume_flip(volume, [self.flip_dim])

        return volume

    def update_coords(self, coords: torch.Tensor) -> torch.Tensor:
        """Apply consistent augmentations to spatial coordinates.

        Transforms coordinates using the same rotation and flipping as applied
        to volumes. This is used for transforming voxel coordinates or object
        bounding box corners.

        Args:
            coords: Tensor of shape [..., 3] containing spatial coordinates (x, y, z).
        Returns:
            Augmented coordinates tensor of shape [..., 3] with transformed spatial
            positions.
        """
        if self.rotate_flag:
            coords = tau.coords_rotz90(coords, self.rotate_time)
        if self.flip_flag:
            coords = tau.coords_flip(coords, [self.flip_dim])

        return coords

    def update_object_bboxes(self, object_bboxes: torch.Tensor) -> torch.Tensor:
        """Apply consistent augmentations to object bounding boxes.

        Transforms bbox coordinates using the same rotation and flipping as
        applied to volumes. After transformation, recalculates min/max corners
        since rotation may swap them.

        Args:
            object_bboxes: Tensor of shape [M, 2, 3] containing M bounding boxes
                where each bbox is [[xmin, ymin, zmin], [xmax, ymax, zmax]].

        Returns:
            Augmented bounding boxes tensor of shape [M, 2, 3] with updated
            coordinates and recomputed min/max corners.

        Notes:
            After rotation or flipping, the original min corner may no longer
            be the minimum, so this method recomputes bbox extrema.

        Examples:
            >>> bboxes = torch.tensor([[[0., 0., 0.], [1., 1., 1.]]])
            >>> augmented_bboxes = aug.update_object_bboxes(bboxes)
            >>> augmented_bboxes.shape
            torch.Size([1, 2, 3])
        """
        object_bboxes = self.update_coords(object_bboxes.view(-1, 3)).view_as(
            object_bboxes
        )
        bbox_mins = torch.min(object_bboxes, dim=1)[0]  # [M, 3]
        bbox_maxs = torch.max(object_bboxes, dim=1)[0]  # [M, 3]
        object_bboxes = torch.stack([bbox_mins, bbox_maxs], dim=1)  # [M, 2, 3]
        return object_bboxes


class Dataset(torch.utils.data.Dataset):
    """PyTorch Dataset for 3D scene completion with sparse voxel representations.

    Loads and processes 3D scenes from multiple datasets (3D-FRONT, ScanNet++, ARKitScenes),
    extracts spatial patches centered on objects, and provides TSDF representations
    with semantic conditioning information.

    Attributes:
        split: Dataset split ("train", "val", or "test").
        data_dict: OrderedDict mapping dataset names to their configurations.
        data_keys: Tuple of data modalities to load (e.g., "tsdf_gt", "mesh_gt").
        augmentation: Whether to apply data augmentation during training.
        voxel_size: Voxel size in meters.
        patch_shape: Spatial dimensions [D, H, W] of extracted patches in voxels.
        chunk_size: Spatial dimensions of patches in world units (meters).
        truncation: TSDF truncation distance in meters.
        known_ratio: Ratio for determining known vs unknown voxels.
        file_paths: List of file path dictionaries for all scenes.
        scene_metas: List of metadata dictionaries for all scenes.
    """

    def __init__(
        self,
        data_list: list[str],
        split: str,
        known_ratio: float,
        patch_shape: Tuple[int, int, int],
        augmentation: bool = False,
        voxel_size: Optional[float] = None,
        truncation: Optional[float] = None,
        data_keys: Optional[Tuple[float, ...]] = None,
        bottom_padding: float = 1.0,
        fixed_len: int = -1,
        ceiling_clip: Optional[float] = None,  # For rendering
        category_include: Optional[str] = None,
        category_exclude: Optional[Tuple[str, ...]] = None,
        bbox_rand_shift: float = 1.0,
        num_categories: int = 100,
        use_semantic_mapping: bool = True,
        scene_names: Optional[Tuple[str, ...]] = None,
        csv_path: Optional[str] = None,
        csv_slice: Optional[Tuple[int, int]] = None,
        patch_names: Optional[str] = None,
        max_scene_volume: Optional[float] = None,
    ):
        """Initialize the 3D scene dataset.

        Args:
            data_list: List of dataset names to include (e.g., ["3D-FRONT", "ScannetPP"]).
            split: Dataset split to load ("train", "val", "test", or "all").
            known_ratio: Ratio threshold for determining known vs unknown regions
                in TSDF. Typically close to 1.0 (e.g., 0.999).
            patch_shape: Spatial dimensions [depth, height, width] of patches in voxels.
            augmentation: If True, apply random rotations and flips during training.
            voxel_size: Size of each voxel in meters (e.g., 0.011 for 1.1cm voxels).
            truncation: TSDF truncation distance in meters. If None, defaults to 3 * voxel_size.
            data_keys: Tuple of data modalities to load (e.g., ("tsdf_gt", "mesh_gt")).
            bottom_padding: Additional padding in voxels below the scene floor.
            fixed_len: If > 0, repeat training data to reach this length. -1 for no repetition.
            ceiling_clip: Height in meters at which to clip ceilings for rendering.
            category_include: Optional category name that must be present in patches.
            category_exclude: Optional tuple of categories to exclude from patches.
            bbox_rand_shift: Random shift range [0, 1] for patch center perturbation.
                1.0 allows shifts up to half an object's size.
            num_categories: Total number of semantic categories in the vocabulary.
            use_semantic_mapping: If True, map raw object names to semantic categories.
        """
        self.patch_names = patch_names
        if patch_names is not None:
            # patch_names: str or tuple of str, each
            # "{scene}_{xmin}_{ymin}_{zmin}" (3 coords, bbox_max from chunk_size)
            # or "{scene}_{xmin}_{ymin}_{zmin}_{xmax}_{ymax}_{zmax}" (6 coords)
            if isinstance(patch_names, str):
                patch_names = (patch_names,)
            self._patches = []
            for pn in patch_names:
                parts = pn.rsplit("_", 6)
                if len(parts) == 7:
                    # 6-coord format: scene + 6 floats
                    self._patches.append((parts[0], [float(x) for x in parts[1:]]))
                else:
                    # 3-coord format fallback
                    parts = pn.rsplit("_", 3)
                    self._patches.append((parts[0], [float(x) for x in parts[1:]]))
            split = "all"  # bypass split filtering
        self.split = split
        self.data_dict = OrderedDict({x: DATA_DICT[x] for x in data_list})
        self.data_keys = data_keys
        self.augmentation = augmentation
        self.fixed_len = fixed_len
        self.ceiling_clip = ceiling_clip
        self.scene_names = scene_names
        self.category_include = category_include
        self.category_exclude = category_exclude
        self.num_categories = num_categories
        self.bottom_padding = bottom_padding
        self.voxel_size = voxel_size
        self.patch_shape = patch_shape
        self.chunk_size = [
            x * voxel_size if x is not None else None for x in patch_shape
        ]
        self.bbox_rand_shift = bbox_rand_shift
        self.known_ratio = known_ratio
        self.truncation = voxel_size * 3 if truncation is None else truncation
        self.use_semantic_mapping = use_semantic_mapping

        self.file_paths, self.scene_metas = self.load_files()

        # Filter out scenes exceeding max volume
        if max_scene_volume is not None:
            keep = []
            for i, sm in enumerate(self.scene_metas):
                bounds = sm["scene_bounds"]  # [6]
                extent = bounds[3:] - bounds[:3]
                volume = float(extent[0] * extent[1] * extent[2])
                if volume <= max_scene_volume:
                    keep.append(i)
                else:
                    logger.info(
                        f"Skipping {sm['scene_name']}: volume {volume:.1f} m³ "
                        f"exceeds max {max_scene_volume:.1f} m³"
                    )
            if len(keep) < len(self.scene_metas):
                logger.info(
                    f"Filtered {len(self.scene_metas) - len(keep)} scenes by volume, "
                    f"{len(keep)} remaining"
                )
                self.file_paths = [self.file_paths[i] for i in keep]
                self.scene_metas = [self.scene_metas[i] for i in keep]

        if patch_names is not None:
            self.csv_rows = []
            for scene_name, coords in self._patches:
                bbox_min = torch.tensor(coords[:3])
                if len(coords) == 6:
                    bbox_max = torch.tensor(coords[3:])
                    bbox = torch.cat([bbox_min, bbox_max])
                elif all(x is not None for x in self.chunk_size):
                    bbox = torch.cat([bbox_min, bbox_min + torch.tensor(self.chunk_size)])
                else:
                    raise ValueError(
                        f"patch_name with 3 coords requires patch_shape to compute bbox size"
                    )
                dataset_name = next(
                    (dn for (dn, sn) in self.scene_lookup if sn == scene_name),
                    None,
                )
                if dataset_name is None:
                    logger.warning(
                        f"Skipping patch {scene_name}: scene not found in any "
                        f"loaded dataset (missing data files?)"
                    )
                    continue
                self.csv_rows.append((dataset_name, scene_name, bbox))
        elif csv_path is not None:
            if not os.path.exists(csv_path):
                raise FileNotFoundError(f"csv_path not found: {csv_path}")
            self.csv_rows = []
            with open(csv_path, newline="") as f:
                for row in csv.DictReader(f):
                    bbox = torch.tensor(
                        [
                            float(row[k])
                            for k in ("xmin", "ymin", "zmin", "xmax", "ymax", "zmax")
                        ]
                    )
                    self.csv_rows.append((row["dataset"], row["scene_name"], bbox))
            if csv_slice is not None:
                start, end = csv_slice
                self.csv_rows = self.csv_rows[start:end]
        else:
            self.csv_rows = None

    def __len__(self) -> int:
        """Return the number of samples in the dataset.

        For training with fixed_len > 0, repeats the dataset to reach the target length.

        Returns:
            Number of samples available in this dataset split.
        """
        if self.csv_rows is not None:
            return len(self.csv_rows)
        if self.fixed_len > 0 and self.split == "train":
            repeat = max(self.fixed_len // len(self.file_paths), 1)
            return len(self.file_paths) * repeat
        else:
            return len(self.file_paths)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single sample from the dataset.

        Args:
            idx: Sample index.

        Returns:
            Dictionary containing the sample data.
        """
        return self.get_sample(idx)

    def load_files(self) -> Tuple[List[Dict[str, str]], List[Dict[str, Any]]]:
        """Load file paths and metadata from all configured datasets.

        Calls load_and_split_data for each dataset in data_list, then interleaves
        the results to mix samples from different datasets.

        Returns:
            Tuple of (file_paths, scene_metas) where:
            - file_paths: List of dicts with file paths for each scene
            - scene_metas: List of dicts with metadata for each scene

        Raises:
            AssertionError: If no valid files are found in any dataset.

        Notes:
            Logs the number of samples loaded from each dataset.
        """
        file_paths, scene_metas = [], []
        self.scene_lookup = {}  # (dataset_name, scene_name) -> (file_path, scene_meta)
        for data_name, data in self.data_dict.items():
            file_paths_i, scene_metas_i = load_and_split_data(
                data_name=data.name,
                root_dir=data.root_dir,
                voxel_size=self.voxel_size,
                data_keys=self.data_keys,
                bottom_padding=self.bottom_padding,
                chunk_size=self.chunk_size,
                overfit_scene=self.scene_names,
                split=self.split,
                fixed_len=self.fixed_len,
                fusion_dir=getattr(data, "fusion_dir", None),
                mapping_path=getattr(data, "mapping_path", None),
                scene_exclude=getattr(data, "scene_exclude", None),
                category_exclude=self.category_exclude,
                category_include=self.category_include,
                use_semantic_mapping=self.use_semantic_mapping,
                keep_region=data.keep_region,
            )
            for fp, sm in zip(file_paths_i, scene_metas_i):
                self.scene_lookup[(data_name, sm["scene_name"])] = (fp, sm)
            file_paths.append(file_paths_i)
            scene_metas.append(scene_metas_i)

        file_paths_ = interleave_lists(file_paths)
        scene_metas_ = interleave_lists(scene_metas)

        assert len(file_paths_) > 0, f"No file found!"

        logger.info(
            f"{self.split.upper()} SAMPLES-{len(file_paths_)}: "
            + "; ".join(
                [
                    f"{data_name}: {len(file_paths_i)}"
                    for data_name, file_paths_i in zip(
                        self.data_dict.keys(), file_paths
                    )
                ]
            )
        )

        return file_paths_, scene_metas_

    def sample_patch_bbox(
        self,
        scene_meta: Dict[str, Any],
        chunk_size: torch.Tensor,
        rng: np.random.Generator,
    ) -> torch.Tensor:
        """Sample a random spatial patch bounding box centered on an object.

        Selects a random object from the scene, perturbs its center position,
        and creates a patch bbox of size chunk_size around that position.

        Args:
            scene_meta: Scene metadata dictionary containing scene_bounds, object_names,
                and object_bboxes.
            rng: Random number generator for reproducible sampling.

        Returns:
            Bounding box tensor of shape [6] as [xmin, ymin, zmin, xmax, ymax, zmax]
            in world coordinates.

        Raises:
            ValueError: If the computed bbox contains NaN values.

        Notes:
            - If category_include is set, only objects of that category are considered
            - Random shift is scaled by bbox_rand_shift (0 = no shift, 1 = full shift)
            - Patch center is clamped to valid region to ensure patch fits in scene
        """

        pos_region = determine_patch_region(scene_meta["scene_bounds"], chunk_size)

        if self.category_include is not None:
            indices = [
                i
                for i, s in enumerate(scene_meta["object_names"])
                if self.category_include == s
            ]
            object_bboxes = scene_meta["object_bboxes"][indices]
            object_volumes = [scene_meta["object_volumes"][i] for i in indices]
        else:
            object_bboxes = scene_meta["object_bboxes"]
            object_volumes = scene_meta["object_volumes"]

        volumes_array = np.array(object_volumes, dtype=np.float64)
        if volumes_array.sum() > 0:
            probabilities = volumes_array / volumes_array.sum()
            object_idx = rng.choice(len(object_bboxes), p=probabilities)
        else:
            object_idx = rng.choice(len(object_bboxes))

        object_bbox = object_bboxes[object_idx]
        object_center = torch.mean(object_bbox, dim=0)  # [3]
        object_size = torch.abs(object_bbox[1] - object_bbox[0])  # [3]

        rand_shift = torch.from_numpy(rng.uniform(-1, 1, size=3)).float()
        rand_shift = rand_shift * object_size / 2.0
        pos_w = object_center + rand_shift * self.bbox_rand_shift
        pos_w = torch.clamp(pos_w, min=pos_region[0], max=pos_region[1])

        bbox_min_world = pos_w - chunk_size / 2.0
        bbox_max_world = pos_w + chunk_size / 2.0
        bbox_world = torch.cat([bbox_min_world, bbox_max_world])  # [6]

        if torch.isnan(bbox_world).any():
            raise ValueError(f"Invalid bbox_world: {bbox_world}")

        return bbox_world

    def get_patch_params(
        self,
        scene_meta: Dict[str, Any],
        bbox_world: torch.Tensor,
        chunk_size: torch.Tensor,
        aug_func: Optional[Augmentations] = None,
    ) -> Optional[Dict[str, Any]]:
        """Extract patch parameters including intersecting objects.

        Finds objects that intersect with the patch bounding box, converts their
        coordinates to patch-local space, and applies augmentations if specified.

        Args:
            scene_meta: Scene metadata containing object names, bboxes, and structure flags.
            bbox_world: Patch bounding box in world coordinates, shape [6].
            aug_func: Optional augmentation function to apply to object bboxes.

        Returns:
            Dictionary with patch parameters:
            - bbox_world: World-space patch bbox
            - voxel_size: Voxel size in meters
            - known_ratio: Known region ratio threshold
            - ceiling_clip: Ceiling clipping height
            - truncation: TSDF truncation distance
            - num_categories: Number of semantic categories
            - object_names: List of intersecting object names
            - object_bboxes: Tensor [M, 2, 3] of object bboxes in patch-local coords
            Returns None if no objects intersect the patch.

        Notes:
            - Structural objects (walls, floors) are clamped to patch boundaries
            - Object bboxes are converted to patch-local coordinates (origin at patch min)
            - If augmentations are applied, bboxes are transformed consistently
        """
        intc_obj_indices = compute_intersected_objects(
            scene_meta["object_bboxes"], bbox_world.tolist()
        )
        if len(intc_obj_indices) == 0:
            return None

        object_bboxes = scene_meta["object_bboxes"][intc_obj_indices]  # [M, 2, 3]
        object_bboxes -= bbox_world[None, None, :3]  # [M, 2, 3]

        if aug_func is not None:
            object_bboxes -= chunk_size[None, None] / 2.0
            object_bboxes = aug_func.update_object_bboxes(object_bboxes)
            object_bboxes += chunk_size[None, None] / 2.0

        object_names = [scene_meta["object_names"][x] for x in intc_obj_indices]
        patch_kwargs = {
            "bbox_world": bbox_world,
            "voxel_size": self.voxel_size,
            "known_ratio": self.known_ratio,
            "ceiling_clip": self.ceiling_clip,
            "truncation": self.truncation,
            "num_categories": self.num_categories,
            "object_names": object_names,
            "object_bboxes": object_bboxes,
        }

        return patch_kwargs

    def get_patch_data(
        self,
        file_paths: Dict[str, str],
        data_key: str,
        bbox_world: torch.Tensor,
        chunk_size: torch.Tensor,
        aug_func: Optional[Augmentations] = None,
    ) -> Optional[Dict[str, Any]]:
        """Load and process a specific data modality for a patch.

        Loads data from disk (TSDF volume or mesh), crops to the patch bbox,
        and converts to the appropriate format (sparse coordinates or mesh).

        Args:
            file_paths: Dictionary mapping data keys to file paths.
            data_key: Name of the data modality to load (e.g., "tsdf_gt", "mesh_gt", "layout_*").
            bbox_world: Patch bounding box in world coordinates, shape [6].
            aug_func: Optional augmentation function for TSDF volumes.

        Returns:
            Dictionary containing patch data with "field" type indicator and data:
            - For "mesh": {"field": "mesh", "mesh": trimesh.Trimesh, "name": data_key}
            - For "layout": {"field": "layout", "name": data_key}
            - For "tsdf": {"field": "tsdf", "band_coords": coords, "band_values": values,
                          "structure": structure_mask, "name": data_key}
            Returns None if no valid TSDF band voxels are found.

        Raises:
            ValueError: If bbox contains non-numeric values or if NaN is detected in tensors.

        Notes:
            - TSDF volumes are clamped to [-truncation, truncation]
            - Only voxels in the truncation band (BAND state) are returned for TSDF
            - Structure mask classifies voxels as EMPTY, UNKNOWN, or BAND
            - Augmentations are applied to TSDF volumes if aug_func is provided
        """
        if data_key == "layout":
            patch = {"field": "layout"}
        elif data_key == "mesh_gt":
            if f"{data_key}_path" in file_paths:
                file_path = file_paths[f"{data_key}_path"]
                mesh = trimesh.load(file_path, process=False)
                mesh.process(validate=True)
                mesh = clip_mesh(mesh, bbox_world.tolist())
                verts = torch.from_numpy(np.array(mesh.vertices, dtype=np.float32))
                verts = verts - bbox_world[None, :3]
                faces = torch.from_numpy(np.array(mesh.faces, dtype=np.int64))
                if aug_func is not None:
                    verts -= chunk_size[None] / 2.0
                    verts = aug_func.update_coords(verts)
                    verts += chunk_size[None] / 2.0
            else:
                verts = torch.empty((0, 3), dtype=torch.float32)
                faces = torch.empty((0, 3), dtype=torch.int64)
            patch = {"field": "mesh", "verts": verts, "faces": faces}
        else:
            patch_shape = torch.round(chunk_size / self.voxel_size).int().tolist()
            file_path = file_paths[f"{data_key}_path"]
            bbox_min = bbox_world.tolist()[:3]
            if not all(isinstance(x, (int, float)) for x in bbox_min):
                raise ValueError(f"Invalid bbox_min values: {bbox_min}")
            volume = loadvdb(file_path)[0]["tsdf"]
            volume = crop_vdb(volume, bbox_min, patch_shape, cell_center=True)
            volume = torch.from_numpy(volume).float()  # [H, W, D]
            volume = torch.clamp(volume, min=-self.truncation, max=self.truncation)
            if aug_func is not None:
                volume = aug_func.update_volume(volume)

            structure = torch.where(
                torch.abs(volume) < self.truncation * self.known_ratio,
                torch.tensor(Voxel.BAND, dtype=torch.int8),
                torch.where(
                    volume >= self.truncation * self.known_ratio,
                    torch.tensor(Voxel.EMPTY, dtype=torch.int8),
                    torch.tensor(Voxel.UNKNOWN, dtype=torch.int8),
                ),
            )

            ijks = torch.nonzero(structure == Voxel.BAND)
            if len(ijks) == 0:
                return None
            values = volume[ijks[:, 0], ijks[:, 1], ijks[:, 2]]

            patch = {
                "field": "tsdf",
                "band_coords": ijks.long().contiguous(),
                "band_values": values[:, None].float().contiguous(),
                "structure": structure[None].contiguous(),
            }

            for key, value in patch.items():
                if isinstance(value, torch.Tensor) and torch.isnan(value).any():
                    raise ValueError(f"NaN detected in {key}")

        return patch

    def get_sample(self, idx: int) -> Dict[str, Any]:
        """Load a complete sample including patch parameters and data modalities.

        Orchestrates the full sample loading pipeline: selects a scene, samples a patch
        bbox, extracts object parameters, and loads requested data modalities.

        Args:
            idx: Sample index in the dataset.

        Returns:
            Dictionary containing:
            - "scene_names": Name of the source scene
            - "field_params": Dictionary with patch parameters (bbox, objects, etc.)
            - Additional keys for each data_key in self.data_keys with loaded data

        Notes:
            - For training with fixed_len, wraps idx to actual dataset length
            - Val/test splits use deterministic RNG (seed=0)
            - Retries with a random sample if no objects intersect or data is empty
            - Augmentations are applied consistently across all data modalities

        Examples:
            >>> dataset = Dataset(data_list=["3D-FRONT"], split="train", ...)
            >>> sample = dataset[0]
            >>> sample.keys()
            dict_keys(['scene_names', 'field_params', 'tsdf_gt'])
        """
        if self.csv_rows is not None:
            dataset_name, scene_name, bbox_world = self.csv_rows[idx]
            key = (dataset_name, scene_name)
            if key not in self.scene_lookup:
                logger.warning(
                    f"CSV entry ({dataset_name!r}, {scene_name!r}) not found in loaded scenes, skipped."
                )
                rng = np.random.default_rng(seed=idx)
                return self.get_sample(rng.integers(0, len(self)))
            file_path, scene_meta = self.scene_lookup[key]
            rng = np.random.default_rng(seed=idx)

            scene_bounds = scene_meta["scene_bounds"]  # [6]
            scene_size = scene_bounds[3:] - scene_bounds[:3]
            chunk_size = [
                scene_size[idx] if a is None else a
                for idx, a in enumerate(self.chunk_size)
            ]
            chunk_size = torch.tensor(chunk_size)
        else:
            if self.fixed_len > 0 and self.split == "train":
                idx = idx % len(self.file_paths)
            if self.split in ["val", "test"]:
                rng = np.random.default_rng(seed=0)
            else:
                rng = np.random.default_rng()
            file_path = self.file_paths[idx]
            scene_meta = self.scene_metas[idx]

            scene_bounds = scene_meta["scene_bounds"]  # [6]
            scene_size = scene_bounds[3:] - scene_bounds[:3]
            chunk_size = [
                scene_size[idx] if a is None else a
                for idx, a in enumerate(self.chunk_size)
            ]
            chunk_size = torch.tensor(chunk_size)
            bbox_world = self.sample_patch_bbox(scene_meta, chunk_size, rng)

        scene_name = scene_meta["scene_name"]
        aug_func = None
        if self.augmentation:
            aug_func = Augmentations(rng)

        field_params = self.get_patch_params(
            scene_meta, bbox_world, chunk_size, aug_func
        )
        if field_params is None:
            logger.info(f"{scene_name}: No intersected objects, retried!")
            return self.get_sample(rng.integers(0, len(self)))

        data = {}
        if self.data_keys is not None:
            data |= {
                k: self.get_patch_data(file_path, k, bbox_world, chunk_size, aug_func)
                for k in self.data_keys
            }

        if any(value is None for value in data.values()):
            logger.debug(f"Empty sample, retried!")
            return self.get_sample(rng.integers(0, len(self)))

        data |= {
            "scene_names": scene_name,
            "field_params": field_params,
            "keep_region": scene_meta["keep_region"],
        }

        return data
