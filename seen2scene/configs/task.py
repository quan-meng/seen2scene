import dataclasses
from typing import Optional, Tuple, Literal


# -----------------------------------------------------------------
# Training / Validation Tasks
# -----------------------------------------------------------------
@dataclasses.dataclass(kw_only=True)
class TrainVAE:
    """Training task configuration for variational autoencoder.

    Defines parameters for training a VAE on 3D scene data, including the
    spatial dimensions and evaluation metrics to track during training.

    Attributes:
        name: Task identifier used for routing and logging.
        scene_shape: 3D spatial dimensions (depth, height, width) of the voxel
            grid processed by the autoencoder. Default (256, 256, 256) provides
            high resolution for scene representation.
        metrics: Tuple of metric names to compute during training. "l1" measures
            mean absolute error, "l2" measures mean squared error between
            reconstructions and targets.
    """

    name: str = "train_vae"
    postfix: str = ""
    scene_shape: Tuple[int, int, int] = (256, 256, 256)
    metrics: Tuple[str, ...] = ("l1", "l2")
    img_wh: int = 150
    backend: Literal["pyrender", "blender"] = "pyrender"


# Train Generator
@dataclasses.dataclass(kw_only=True)
class TrainGen:
    """Training task configuration for flow matching generator.

    Defines parameters for training a flow matching model to generate 3D scenes,
    including sampling parameters, rendering settings, and evaluation metrics.

    Attributes:
        name: Task identifier used for routing and logging.
        num_variations: Number of generation variations to produce per input during
            evaluation. Multiple variations help assess generation diversity.
        num_for_plot: Number of samples to visualize in plots and logging.
            Lower number reduces logging overhead.
        num_samples: Total number of samples to generate for evaluation metrics.
            Larger values provide more robust metric estimates.
        num_views_for_fid: Number of rendered camera views per scene for computing
            Fréchet Inception Distance (FID). More views capture scene quality better
            but increase computation.
        guidance_scale: Classifier-free guidance scale controlling adherence to
            conditioning. Higher values (e.g., 3.0) increase conditioning strength
            but may reduce diversity.
        backend: Rendering backend for visualization. "pyrender" is faster for
            development, "blender" provides higher quality renders.
        scene_shape: 3D spatial dimensions (depth, height, width) of the generated
            voxel grid.
        metrics: Tuple of metric names to compute. "fid" measures the perceptual
            quality and diversity of generated scenes.
    """

    name: str = "train_gen"
    postfix: str = ""
    num_variations: int = 1
    num_for_plot: int = 1
    num_samples: int = 128
    img_wh: int = 150
    num_views_for_fid: int = 10
    guidance_scale: float = 3.0
    backend: Literal["pyrender", "blender"] = "pyrender"
    scene_shape: Tuple[int, int, int] = (256, 256, 256)
    metrics: Tuple[str, ...] = ("fid",)


@dataclasses.dataclass(kw_only=True)
class TrainControl(TrainGen):
    """Training task configuration for ControlNet generator.

    Extends TrainGen with the same parameters, but used for training a ControlNet
    architecture instead of a flow matching model. Inherits all training settings
    from parent, only overriding the task name.

    Attributes:
        name: Task identifier for ControlNet training mode.
        All other attributes are inherited from TrainGen.
    """

    name: str = "train_control"
    num_variations: int = 1
    repeat_num: int = 1
    num_samples: int = 8
    metrics: Tuple[str, ...] = ("iou", "l1", "l2", "tmd")
    drop_bbox: bool = False


@dataclasses.dataclass(kw_only=True)
class ValCompletion(TrainControl):
    """Validation task for scene completion using the val split.

    Runs validation_step on the validation set to evaluate completion quality,
    generating multiple variations per sample to assess diversity via TMD.

    Attributes:
        name: Task identifier for validation completion mode.
        num_variations: Number of variations to generate per sample for diversity metrics.
        metrics: Default metric is TMD (Total Mutual Difference) for diversity evaluation.
    """

    name: str = "val_completion"
    num_variations: int = 3
    overwrite: bool = False
    metrics: Tuple[str, ...] = ("tmd",)


# -----------------------------------------------------------------
# Applications
# -----------------------------------------------------------------
@dataclasses.dataclass(kw_only=True)
class Reconstruction:
    """Reconstruction task configuration for autoencoder inference.

    Defines parameters for running VAE reconstruction on test data, including
    data split selection, export formats, and evaluation metrics.

    Attributes:
        name: Task identifier for reconstruction mode.
        split: Dataset split to use for reconstruction. Default "test" evaluates
            on held-out test data.
        num_samples: Number of samples to reconstruct and evaluate.
        export_as: Tuple of export formats for saving reconstructions. Options:
            - "tree": Export as octree structure for efficient sparse representation
            - "npz": Export as NumPy compressed arrays for analysis
        metrics: Tuple of metric names to compute. "l1" and "l2" measure
            reconstruction quality against ground truth.
    """

    name: str = "reconstruction"
    postfix: str = ""
    split: str = "test"
    num_samples: int = 10
    export_as: Tuple[str, ...] = ("mesh", "npz", "volume")  # "npz"
    metrics: Tuple[str, ...] = ("l1", "l2")
    scene_shape: Tuple[int, int, int] = (256, 256, 256)


@dataclasses.dataclass(kw_only=True)
class Generation:
    """Generation task configuration for unconditional scene synthesis.

    Defines parameters for generating complete 3D scenes from scratch using
    the trained generator, including rendering, export, and evaluation settings.

    Attributes:
        name: Task identifier for patch-based generation mode.
        num_samples: Number of scenes to generate for evaluation.
        img_wh: Image width and height in pixels for rendered views. Higher
            resolution (e.g., 512) provides better quality for FID computation.
        export_as: Tuple of export formats for generated scenes. Options:
            - "mesh": Export as triangle mesh for visualization and downstream use
            - "bbox": Export bounding box information for spatial analysis
            - "npz": Export as NumPy arrays (optional)
        num_views_for_fid: Number of rendered camera views per scene for FID metric.
            More views provide better coverage of scene appearance.
        repeat_num: Number of times to repeat generation for each sample. Useful
            for analyzing generation stochasticity.
        guidance_scale: Classifier-free guidance scale. Higher values increase
            adherence to any conditioning but may reduce diversity.
        backend: Rendering backend for visualization. "pyrender" is faster,
            "blender" provides photorealistic quality.
        scene_shape: 3D spatial dimensions (depth, height, width) of generated
            voxel grid.
        vis_latent_pca: If True, visualize PCA projection of latent codes for analysis
            of latent space structure.
        metrics: Tuple of metric names to compute. "fid" evaluates perceptual
            quality and diversity of generations.
    """

    name: str = "patch_generation"
    postfix: str = ""
    num_samples: int = 1000
    img_wh: int = 512
    export_as: Tuple[str, ...] = ("bbox", "mesh")  # "npz"
    num_views_for_fid: int = 10
    repeat_num: int = 1
    num_variations: int = 1
    guidance_scale: float = 3.0
    backend: Literal["pyrender", "blender"] = "pyrender"
    scene_shape: Tuple[int, int, int] = (256, 256, 256)
    overwrite: bool = False
    metrics: Tuple[str, ...] = ("fid",)

    # debug
    vis_latent_pca: bool = False
    vis_rope_pca: bool = False  # Visualize CLIP embeddings with distance RoPE using PCA
    vis_sparity: bool = False
    vis_scan_prob: bool = False  # Export scan probability as heatmap point cloud


@dataclasses.dataclass(kw_only=True)
class Completion(Generation):
    """Completion task configuration for conditional scene completion.

    Extends Generation for scene completion tasks where partial observations
    are completed to full scen
    es. Inherits all generation parameters from parent.

    Attributes:
        name: Task identifier for completion mode.
        metrics: Evaluation metrics for scene completion: IoU, L1/L2 error,
            and Total Mutual Difference (TMD) for diversity.
    """

    name: str = "patch_completion"
    export_volume: bool = True
    export_as: Tuple[str, ...] = ("bbox", "mesh", "volume")
    drop_bbox: bool = False
    metrics: Tuple[str, ...] = ("iou", "l1", "l2", "tmd")


@dataclasses.dataclass(kw_only=True)
class TextToScene(Generation):
    """Layout editing task configuration for scene modification.

    Extends Generation for tasks involving editing spatial layouts of existing
    scenes. Loads a scene layout from the dataset, uses an LLM (Codex/Claude)
    to modify the layout based on a text prompt, then generates geometry for
    the edited layout.

    Attributes:
        name: Task identifier for layout editing mode.
        prompt: Natural language instruction for how to edit the layout
            (e.g., "add a computer on the table", "remove the nightstand").
    """

    name: str = "text2scene"
    prompt: str = ""
    scene_shape: Tuple[Optional[int], Optional[int], int] = (None, None, 256)


@dataclasses.dataclass(kw_only=True)
class Image2Scene(Generation):
    """Image-to-scene generation task configuration.

    Extends Generation for tasks that generate 3D scenes conditioned on input
    images. Inherits all generation parameters from parent.

    Attributes:
        name: Task identifier for image-to-scene generation mode.
    """

    name: str = "image2scene"


# Large Scale -------------------------------------------------
@dataclasses.dataclass(kw_only=True)
class LargeScaleGeneration(Generation):
    """Large-scale generation task for synthesizing extended 3D scenes.

    Extends Generation to handle larger scene dimensions that exceed the model's
    native resolution. Uses a sliding window approach with overlapping patches
    to generate scenes of arbitrary size while maintaining coherence.

    Attributes:
        name: Base task identifier, will be augmented with scene dimensions.
        scene_shape: 3D spatial dimensions (depth, height, width) for large-scale
            generation. Default (384, 384, 256) is 1.5x larger than standard.
        overlap: Overlap ratio between adjacent patches in sliding window generation.
            0.2 means 20% overlap, which helps blend patches and reduce seams.

    Notes:
        The __post_init__ method appends scene dimensions to the task name,
        creating identifiers like "large_scale_generation_384_384_256".
    """

    name: str = "large_scale_generation"
    scene_shape: Tuple[Optional[int], Optional[int], int] = (None, None, 256)
    overlap: float = 0.2
    cpu_offload: bool = True
    max_scene_volume: float = (
        70.0  # Skip scenes larger than this volume (m³), default 5*5*2.8
    )


@dataclasses.dataclass(kw_only=True)
class LargeScaleTextToScene(LargeScaleGeneration):
    """Layout editing task configuration for scene modification.

    Extends Generation for tasks involving editing spatial layouts of existing
    scenes. Loads a scene layout from the dataset, uses an LLM (Codex/Claude)
    to modify the layout based on a text prompt, then generates geometry for
    the edited layout.

    Attributes:
        name: Task identifier for layout editing mode.
        prompt: Natural language instruction for how to edit the layout
            (e.g., "add a computer on the table", "remove the nightstand").
    """

    name: str = "large_scale_text2scene"


@dataclasses.dataclass(kw_only=True)
class LargeScaleCompletion(Generation):
    """Large-scale completion task for completing extended partial scenes.

    Extends Generation to handle scene completion at larger dimensions using
    a sliding window approach. Useful for completing large-scale partial
    observations like LiDAR scans of building interiors or outdoor environments.

    Attributes:
        name: Base task identifier, will be augmented with scene dimensions.
        scene_shape: 3D spatial dimensions (depth, height, width) for large-scale
            completion. Default (384, 384, 256) provides rectangular volume.
        overlap: Overlap ratio between adjacent patches in sliding window completion.
            0.2 means 20% overlap for smooth blending across patch boundaries.

    Notes:
        The __post_init__ method appends scene dimensions to the task name,
        creating identifiers like "large_scale_completion_384_384_256".
    """

    name: str = "large_scale_completion"
    scene_shape: Tuple[Optional[int], Optional[int], int] = (None, None, 256)
    metrics: Tuple[str, ...] = ()
    overlap: float = 0.2
    drop_bbox: bool = False
    cpu_offload: bool = True
    max_scene_volume: float = (
        70.0  # Skip scenes larger than this volume (m³), default 5*5*2.8
    )
