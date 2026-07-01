import dataclasses
import tyro
from typing import *
from dataclasses import field
from enum import IntEnum

from seen2scene import (
    ARKITSCENES_DIR,
    ARKITSCENES_LIDAR_DIR,
    ARKITSCENES_RAW_DIR,
    ARKITSCENES_TMP_DIR,
    ASSETS_DIR,
    FRONT3D_DIR,
    FRONT3D_RAW_DIR,
    LLMSLAYOUT_DIR,
    SCANNETPP_DIR,
    SCANNETPP_LIDAR_DIR,
    SCANNETPP_RAW_DIR,
)


class Voxel(IntEnum):
    """Enumeration of voxel occupancy states in 3D scene representations.

    Defines the semantic categories for voxel occupancy in TSDF (Truncated Signed
    Distance Function) representations. Each voxel in the grid is assigned one
    of these states to indicate its relationship to scene geometry.

    Attributes:
        EMPTY: Voxel is definitively empty space (value 0). No surface passes
            through this region based on observations.
        UNKNOWN: Voxel occupancy is unknown (value 1). Typically occurs in
            unobserved regions or areas outside sensor range.
        BAND: Voxel is in the truncation band near surfaces (value 2). Contains
            valid TSDF values indicating distance to nearest surface.
    """

    EMPTY = 0
    UNKNOWN = 1
    BAND = 2


@dataclasses.dataclass(kw_only=True)
class Front3D:
    """Configuration for 3D-FRONT synthetic indoor scene dataset.

    Specifies paths and parameters for the 3D-FRONT dataset, which contains
    synthetic indoor scenes with furniture and room layouts. Includes both
    raw data paths (3D models, textures) and processed data paths (voxelized
    scenes, depth renderings).

    Attributes:
        name: Dataset identifier used for logging and data routing.
        future_path: Directory containing 3D-FUTURE furniture model assets.
        texture_path: Directory containing texture maps for 3D-FRONT scenes.
        raw_dir: Root directory for raw 3D-FRONT dataset files.
        json_dir: Directory containing JSON scene layout files.
        cctextures_dir: Directory containing CC Textures for material synthesis.
        min_depth: Minimum depth value in meters for depth rendering. Closer
            pixels are clipped to this value.
        max_depth: Maximum depth value in meters for depth rendering. Further
            pixels are clipped to this value.
        root_dir: Root directory for processed dataset (voxelized scenes).
        fusion_dir: Dictionary mapping voxel size strings to fusion output
            directories. Different voxel sizes may be stored on different storage
            systems for space efficiency.
        mapping_path: Path to CSV file mapping object categories to semantic labels.
        scan_name: Identifier for the synthetic scan configuration, encoding
            FOV (60°), distance (8.0m), and resolution (512px).
        scene_exclude: Tuple of scene IDs to exclude from training/evaluation.
            Excluded scenes may have data quality issues (e.g., multiple floors).
    """

    # Raw data -------------------------------------------------------------------
    name: str = "3D-FRONT"
    future_path: str = str(FRONT3D_RAW_DIR / "3D-FUTURE-model")
    texture_path: str = str(FRONT3D_RAW_DIR / "3D-FRONT-texture")
    raw_dir: str = str(FRONT3D_RAW_DIR)
    json_dir: str = str(FRONT3D_RAW_DIR / "3D-FRONT")
    cctextures_dir: str = str(FRONT3D_RAW_DIR / "cctextures")
    min_depth: float = 0.0  # in meters
    max_depth: float = 100.0  # in meters

    # Processed data --------------------------------------------------------------
    root_dir: str = str(FRONT3D_DIR)
    fusion_dir: Dict[str, str] = field(
        default_factory=lambda: {"0.011": str(FRONT3D_DIR)}
    )
    mapping_path: str = str(ASSETS_DIR / "front3d_mapping.csv")
    scan_name: str = "scans_fov_60.0_d_8.0_r_512"
    keep_region: Optional[str] = "obj_interior"
    scene_exclude: Optional[Tuple[str, ...]] = (
        "979a4a2f-32c4-42fa-b078-7466c093bcf2",  # 2 floors
    )


@dataclasses.dataclass(kw_only=True)
class ScannetPP:
    """Configuration for ScanNet++ real-world indoor scene dataset.

    Specifies paths and parameters for the ScanNet++ dataset, which contains
    high-quality RGB-D scans and LiDAR captures of real indoor environments.
    Provides denser and higher-quality data than the original ScanNet dataset.

    Attributes:
        name: Dataset identifier used for logging and data routing.
        raw_dir: Directory containing official ScanNet++ v2 raw data including
            RGB-D sequences and camera poses.
        lidar_dir: Directory containing high-resolution LiDAR point cloud scans
            for ScanNet++ scenes.
        scene_list_path: Path to CSV file listing available scenes and their metadata,
            mapping scene release IDs to Faro timestamps.
        root_dir: Root directory for processed dataset (voxelized fusion results).
        fusion_dir: Dictionary mapping voxel size strings to fusion output
            directories. ScanNet++ supports multiple voxel resolutions for
            different use cases.
        mapping_path: Path to CSV file mapping object categories to semantic labels
            for the processed data.
        scene_exclude: Optional tuple of scene IDs to exclude. None means no
            exclusions for ScanNet++ (all scenes have good quality).
    """

    # Raw data -------------------------------------------------------------------
    name: str = "ScannetPP"
    raw_dir: str = str(SCANNETPP_RAW_DIR)
    lidar_dir: str = str(SCANNETPP_LIDAR_DIR)

    # Processed data --------------------------------------------------------------
    scene_list_path: str = "PATH_TO_SCANNETPP_SCENE_MAPPING_CSV"
    root_dir: str = str(SCANNETPP_DIR)
    fusion_dir: Dict[str, str] = field(
        default_factory=lambda: {"0.011": str(SCANNETPP_DIR)}
    )
    mapping_path: str = str(ASSETS_DIR / "scannetpp_mapping.csv")
    scene_exclude: Optional[Tuple[str, ...]] = None
    keep_region: Optional[str] = "obj_interior"


@dataclasses.dataclass(kw_only=True)
class ARKitScenes:
    """Configuration for ARKitScenes mobile device capture dataset.

    Specifies paths and parameters for the ARKitScenes dataset, which contains
    3D scans captured using Apple's ARKit on mobile devices. Includes both
    RGB-D sequences and high-quality LiDAR ground truth.

    The dataset has significant data quality variations due to mobile capture
    conditions, leading to a large exclusion list of scenes with misaligned
    LiDAR or other quality issues.

    Attributes:
        name: Dataset identifier used for logging and data routing.
        raw_dir: Root directory containing raw ARKitScenes dataset files.
        lidar_dir: Directory containing laser scanner point clouds providing
            high-quality ground truth geometry.
        tmp_dir: Temporary directory for intermediate processing outputs before
            final fusion results.
        root_dir: Root directory for final processed dataset (voxelized fusion).
        fusion_dir: Dictionary mapping voxel size strings to fusion output
            directories. ARKitScenes currently supports 0.011m voxel size.
        mapping_path: Path to CSV file mapping object categories to semantic labels.
        scene_exclude: Tuple of 76 scene IDs excluded due to data quality issues,
            primarily LiDAR misalignment with RGB-D captures.
    """

    # Raw data -------------------------------------------------------------------
    name: str = "ARKitScenes"
    raw_dir: str = str(ARKITSCENES_RAW_DIR)
    lidar_dir: str = str(ARKITSCENES_LIDAR_DIR)

    # intermediate data -----------------------------------------------------------
    tmp_dir: str = str(ARKITSCENES_TMP_DIR)

    # Processed data --------------------------------------------------------------
    root_dir: str = str(ARKITSCENES_DIR)
    fusion_dir: Dict[str, str] = field(
        default_factory=lambda: {
            "0.011": str(ARKITSCENES_DIR)
        }
    )
    mapping_path: str = str(ASSETS_DIR / "arkitscenes_mapping.csv")
    keep_region: Optional[str] = "obj_interior"
    scene_exclude: Optional[Tuple[str, ...]] = (
        "421006",
        "421009",
        "421012",
        "421016",
        "421252",
        "421255",
        "421256",
        "421259",
        "421260",
        "421337",
        "421392",
        "421655",
        "421659",
        "422022",
        "422214",
        "422382",
        "422543",
        "423980",
        "435342",
        "435368",
        "435654",
        "435659",
        "435730",
        "437126",
        "437135",
        "437253",
        "437298",
        "464746",
        "464789",
        "464803",
        "466234",
        "466846",
        "466849",
        "468298",
        "468649",
        "468775",
        "469452",
        "469635",
        "469819",
        "469837",
        "470098",
        "470102",
        "470136",
        "470341",
        "471442",
        "472026",
        "472041",
        "472042",
        "472050",
        "472058",
        "472078",
        "472092",
        "472201",
        "472307",
        "472308",
        "472312",
        "472316",
        "472326",
        "472335",
        "472347",
        "472356",
        "472475",
        "472477",
        "472484",
        "472591",
        "472595",
        "472603",
        "472621",
        "472625",
        "472626",
        "481488",
        "481511",
        "481529",
        "483089",
        "483591",
        "484003",  # Lidars misaligned: 76 scenes
    )


class LLMsLayout:
    """Configuration for LLMsLayout dataset.

    Specifies paths and parameters for the LLMsLayout dataset, which contains
    layout data generated or processed using large language models (LLMs).

    Attributes:
        name: Dataset identifier used for logging and data routing.
        root_dir: Root directory containing LLMsLayout dataset files.
        keep_region: Optional region filter (None means use full scene).
    """

    name: str = "LLMsLayout"
    root_dir: str = str(LLMSLAYOUT_DIR)
    keep_region: Optional[str] = None


"""Dictionary mapping dataset names to their configuration instances.

Provides centralized access to dataset configurations by name. Each entry
contains a pre-initialized configuration dataclass with default paths and
parameters for that dataset.

Keys:
    "3D-FRONT": Synthetic indoor scenes with furniture
    "ScannetPP": High-quality real indoor RGB-D + LiDAR scans
    "ARKitScenes": Mobile-captured indoor scenes with ARKit
"""
DATA_DICT: Dict[str, Union[Front3D, ScannetPP, ARKitScenes]] = {
    "3D-FRONT": Front3D(),
    "ScannetPP": ScannetPP(),
    "ARKitScenes": ARKitScenes(),
    "LLMsLayout": LLMsLayout(),
}


@dataclasses.dataclass(kw_only=True)
class Dataset:
    """Main dataset configuration for training and evaluation.

    Aggregates configuration parameters for data loading, augmentation, voxelization,
    and semantic processing. Supports multi-dataset training by combining multiple
    sources (synthetic and real scans).

    Attributes:
        target: Python import path to the Dataset class implementation.
        data_list: Tuple of dataset names to include in training. Datasets are
            sampled uniformly during training.
        augmentation: If True, apply data augmentation (rotation, scaling, jittering)
            during training.
        fixed_len: Fixed sequence length for point cloud batching. -1 disables padding
            and uses variable-length sequences. Padding ensures consistent batch sizes.
        known_ratio: Ratio of voxels to mark as "known" during partial scene training.
            0.999 means 99.9% of voxels are observed, used for scene completion tasks.
        voxel_size: Size of voxels in meters. 0.011m (~1.1cm) provides good balance
            between resolution and memory usage.
        truncation: TSDF truncation distance in meters. If None, defaults to
            3 * voxel_size. Determines the band width around surfaces.
        data_keys: Tuple of data modalities to load (e.g., "tsdf", "latent", "rgb").
            Configured by task-specific settings.
        patch_shape: 3D dimensions (depth, height, width) of spatial patches to
            extract from scenes. None uses full scenes.
        bottom_padding: Additional padding in meters added below scenes to include
            floor regions that may be partially cropped.
        bbox_rand_shift: Random shift range [0, 1] for bounding box jittering during
            augmentation. 1.0 allows shifts up to full bbox dimensions.
        num_categories: Total number of semantic categories in the label space.
        ceiling_clip: Height in meters at which to clip ceilings for rendering.
            Prevents overly tall rooms from affecting camera placement.
        category_exclude: Tuple of category names to exclude from semantic conditioning.
            Structural elements like walls/floors are excluded as they're ubiquitous.
        category_include: Optional specific category name to include exclusively.
            If set, only this category is used for conditioning.
        use_semantic_mapping: If True, map raw object names to standardized semantic
            categories. If False, use raw names directly.
        scene_names: Optional tuple of scene IDs to load exclusively, bypassing
            the train/val/test split. Useful for debugging specific scenes.
    """

    target: str = "seen2scene.dataset.dataset.Dataset"
    data_list: Tuple[str, ...] = ("3D-FRONT", "ScannetPP", "ARKitScenes")
    patch_shape: Annotated[Optional[Tuple[int, int, int]], tyro.conf.Suppress] = None
    augmentation: bool = True
    fixed_len: int = 20000  # -1 not padding
    known_ratio: float = 0.999
    voxel_size: Optional[float] = 0.011
    truncation: Optional[float] = None  # if None: 3 * voxel_size
    bottom_padding: float = 3.0
    bbox_rand_shift: float = 1.0
    num_categories: int = 150
    ceiling_clip: float = 2.2  # For rendering
    category_exclude: Tuple[str, ...] = ("floor", "ceiling", "wall")
    category_include: Optional[str] = None
    use_semantic_mapping: bool = False
    max_scene_volume: Optional[float] = None  # Skip scenes exceeding this volume (m³)
    csv_path: Optional[str] = None  # For deterministic patch sampling in eval
    csv_slice: Optional[Tuple[int, int]] = None  # (start_idx, end_idx) row range
    patch_names: Optional[Union[str, Tuple[str, ...]]] = (
        None  # e.g. "478011_2.2_-1.2_76.1"; bypasses split
    )
    scene_names: Optional[Tuple[str, ...]] = None

    def __post_init__(self) -> None:
        """Validate dataset configuration after initialization.

        Automatically called after dataclass construction. Performs validation
        checks to ensure configuration parameters are in valid ranges.

        Raises:
            AssertionError: If bbox_rand_shift is not in the range [0.0, 1.0].
                Values outside this range would produce invalid bbox perturbations.
        """
        assert (
            0.0 <= self.bbox_rand_shift <= 1.0
        ), "bbox_rand_shift must be between 0.0 and 1.0"
