import os
import sys
import yaml
import dataclasses
from typing import *
from datetime import datetime
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from dataclasses import field

from .dataset import *
from .autoencoder import *
from .generator import *
from .task import *
from seen2scene import EXP_DIR
from seen2scene.tools.common_utils import float2str
from seen2scene.tools.log_utils import get_logger
from seen2scene.tools.slurm_utils import Slurm
from seen2scene.tools.log_utils import create_slurm_job_name_from_tyro

logger = get_logger(file_name=__file__, debug="opt")


@dataclasses.dataclass(kw_only=True)
class LRScheduler:
    """Learning rate scheduler configuration.

    Defines parameters for learning rate scheduling during training, including
    warmup steps and learning rate bounds.

    Attributes:
        name: Name of the learning rate scheduler strategy. Default is "cosine_warmup"
            which applies cosine annealing with warmup.
        num_warmup_steps: Number of initial training steps for learning rate warmup.
            The learning rate gradually increases from 0 to lr during this period.
        lr: Initial (maximum) learning rate after warmup phase.
        min_lr: Minimum learning rate floor to prevent lr from decaying below this value.
    """

    name: str = "cosine_warmup"
    num_warmup_steps: int = 1000.0
    lr: float = 1e-4
    min_lr: float = 1e-7


@dataclasses.dataclass(kw_only=True)
class Base:
    """Base configuration class for training experiments.

    Provides common configuration parameters for PyTorch Lightning training,
    including SLURM job management, data loading, optimization, logging,
    and distributed training settings. Serves as the parent class for
    specific model configurations like AutoEncoder and Generator.

    Attributes:
        slurm: SLURM job submission configuration for cluster computing.
        data: Dataset configuration specifying data sources and loading parameters.
        lr_scheduler: Learning rate scheduler configuration.
        num_workers: Number of subprocesses for data loading. Higher values can
            improve data loading throughput but consume more memory.
        benchmark: Enable CUDNN benchmark mode for faster runtime when input sizes
            are fixed. Should be disabled for variable input sizes.
        precision: Training precision format. Options: "16" (FP16), "bf16" (BFloat16),
            "16-mixed" (mixed FP16), "bf16-mixed" (mixed BF16), "32" (FP32).
        strategy: Distributed training strategy. Default "ddp" for DistributedDataParallel.
        accelerator: Hardware accelerator type. Options: "gpu", "cpu".
        seed: Random seed for reproducibility across runs.
        profiler: PyTorch Lightning profiler type. Options: 'advanced', 'simple', or None.
        log_root: Root directory for all experiment logs.
        log_dir: Directory for current experiment logs, created within log_root.
        slurm_folder: Directory for SLURM job scripts and outputs.
        job_name: Human-readable name for the training job.
        timestamp: Timestamp string for the current run, auto-generated in format
            YYYY-MM-DD_HH-MM-SS-mmm.
        max_epochs: Maximum number of training epochs. Training stops when reached.
        max_steps: Maximum number of training steps. -1 means no step limit.
        resume: Path to checkpoint for resuming training. None starts from scratch.
        learning_rate: Initial learning rate for the optimizer.
        train_data_keys: Tuple of data keys to load from training dataset.
        val_data_keys: Tuple of data keys to load from validation dataset.
        test_data_keys: Tuple of data keys to load from test dataset.
        ema_update_batch: Frequency of EMA (Exponential Moving Average) updates in batches.
            Stable Diffusion 3 uses 100, original SD uses 1.
        check_val_every_n_epoch: Run validation every N epochs.
        accumulate_grad_batches: Number of batches to accumulate gradients before
            optimizer step. Effectively multiplies batch size.
        num_sanity_val_steps: Number of validation steps to run before training starts,
            for sanity checking the validation loop.
        gradient_clip_val: Maximum gradient norm (if algorithm="norm") or maximum
            gradient value (if algorithm="value"). None disables clipping.
        gradient_clip_algorithm: Gradient clipping method. Options: None, "norm", "value".
        monitor_gradients: Enable gradient and parameter monitoring for diagnostics.
        monitor_grad_freq: Log gradient statistics every N training steps.
        monitor_grad_histograms: Log full histograms for detailed analysis (slower).
        val_split: Name of validation split to use from the dataset configuration.
        test_split: Name of test split to use from the dataset configuration.
        batch_size: Batch size
        latent_key: Key name for accessing latent representations in the dataset.
        mask_unknown: If True, mask unknown/incomplete regions during training.
        ckpt_path: Path to checkpoint file for model initialization.
        ignore_keys: Tuple of parameter keys to ignore when loading from checkpoint.
    """

    slurm: Slurm
    data: Dataset
    lr_scheduler: LRScheduler

    num_workers: int = 4  # Number of subprocesses for data loading
    benchmark: bool = True  # Enable cudnn benchmark for faster runtime

    precision: str = "32"  # ["16", "bf16", "16-mixed", "bf16-mixed", "32"]
    strategy: str = "ddp_find_unused_parameters_true"  # or "ddp"
    accelerator: str = "gpu"  # ["gpu", "cpu"]

    seed: int = 23
    profiler: Optional[str] = "simple"  # ['advanced', 'simple', None]
    log_root: str = str(EXP_DIR)
    log_dir: str = "./log"
    slurm_folder: str = "./log"
    job_name: str = "unnamed_job"
    timestamp: str = ""
    max_epochs: int = 300  # Maximum number of epochs to train
    max_steps: int = -1  # Maximum number of steps to train (-1 for no limit)
    resume: Optional[str] = None  # resume training
    learning_rate: float = 1.0e-4  # Initial learning rate for optimization
    batch_size: int = 16

    val_split: str = "val"
    test_split: str = "test"
    train_data_keys: Optional[Tuple[str, ...]] = None
    val_data_keys: Optional[Tuple[str, ...]] = None
    test_data_keys: Optional[Tuple[str, ...]] = None

    ema_update_batch: int = 5  # SD3: 100, SD: 1
    check_val_every_n_epoch: int = 5  # Frequency of validation in epochs
    accumulate_grad_batches: int = 1  # x batch
    num_sanity_val_steps: int = 1
    gradient_clip_val: Optional[float] = 1.0  # None, 1.0 for norm, 0.5 for value
    gradient_clip_algorithm: Optional[str] = "norm"  # None, "norm", "value"

    # Gradient and parameter monitoring for training diagnostics
    # Logs global statistics and histograms (not per-layer to avoid metric explosion)
    monitor_gradients: bool = False  # Enable gradient/parameter monitoring
    monitor_grad_freq: int = (
        50  # Log gradient stats every N steps (10 for debug, 100 for tuning, 500 for production)
    )
    monitor_grad_histograms: bool = (
        False  # Reserved (histograms always logged to TensorBoard when available)
    )

    # Semantic class distribution monitoring
    monitor_semantic_distribution: bool = (
        False  # Enable semantic class distribution monitoring
    )
    monitor_semantic_freq: int = 50  # Log class distribution every N steps
    monitor_semantic_top_k: int = 50  # Number of top classes to show in plots

    latent_key: Optional[str] = None
    mask_unknown: bool = True
    ckpt_path: Optional[str] = None
    ignore_keys: Optional[Tuple[str, ...]] = None

    def update_slurm_config(self) -> None:
        """Update SLURM job configuration based on training parameters.

        Generates a descriptive job name from command-line arguments, configures
        CPU allocation, and sets up the SLURM output folder. The job name is
        truncated to 100 characters to comply with SLURM naming limits.

        The job name is created by parsing sys.argv and excluding parameters that
        don't meaningfully distinguish different experiments (like paths and sizes).

        Notes:
            Modifies the following attributes in-place:
            - self.slurm.cpus_per_task: Set to match num_workers
            - self.slurm.slurm_job_name: Set to timestamped job name
            - self.job_name: Set to same timestamped job name
            - self.slurm_folder: Set to log directory path
        """
        skip_params = [
            "slurm",
            "log_dir",
            "resume",
            "ckpt_path",
            "patch_shape",
            "batch_size",
            "logdir_ae",
        ]
        job_name = create_slurm_job_name_from_tyro(sys.argv, skip_params=skip_params)
        self.slurm.cpus_per_task = self.num_workers
        self.slurm.slurm_job_name = self.job_name = f"{self.timestamp}_{job_name}"[:100]
        self.slurm_folder = self.log_dir
        logger.info(f"Job name: {self.job_name}")

    def resume_or_from_sratch(self, out_dir: str) -> None:
        """Configure experiment directory for resuming or starting from scratch.

        Determines whether to resume training from an existing checkpoint or start
        a new training run. Sets up the log directory and timestamp accordingly.

        For resuming: Validates that the checkpoint directory exists and contains
        a valid checkpoint folder, then configures paths to use the existing experiment.

        For new training: Generates a unique timestamp and creates a new log directory
        under the specified output directory.

        Args:
            out_dir: Base output directory where experiment folders are stored.
                For new runs, a timestamped subdirectory will be created here.
                For resumed runs, this should contain the named checkpoint folder.

        Raises:
            AssertionError: If resume path contains '/' (should be a simple folder name),
                if the resume path doesn't exist, or if the checkpoint folder is missing.

        Notes:
            Modifies the following attributes:
            - self.resume: Converted to absolute path if resuming
            - self.log_dir: Set to resume path or new timestamped directory
            - self.timestamp: Extracted from resume path or newly generated
        """
        if self.resume is not None:
            assert (
                "/" not in self.resume
            ), f"resume path should not contain /: {self.resume}"
            self.resume = os.path.join(out_dir, self.resume)
            assert os.path.exists(self.resume), f"{self.resume} does not exist!"
            assert os.path.exists(os.path.join(self.resume, "checkpoint")), logger.info(
                f"{self.resume} does not contain checkpoint!"
            )
            self.log_dir = self.resume
            self.timestamp = os.path.basename(self.resume)
            logger.info(f"Resume from {self.resume}")
        else:
            self.timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")[:-3]
            self.log_dir = os.path.join(out_dir, self.timestamp)
            os.makedirs(self.log_dir, exist_ok=True)
            logger.info(f"Run from scratch in: {self.log_dir}")

    def update_data_config(self) -> None:
        """Update dataset configuration based on task requirements.

        Synchronizes data loading keys across train/val/test splits and sets
        the patch shape to match the task's scene shape. This ensures consistent
        data handling across all dataset splits.

        Notes:
            Modifies the following attributes:
            - self.train_data_keys: Set to [self.latent_key]
            - self.val_data_keys: Set to [self.latent_key]
            - self.test_data_keys: Set to [self.latent_key]
            - self.data.patch_shape: Set to self.task.scene_shape
        """
        self.train_data_keys = [self.latent_key]
        self.val_data_keys = self.test_data_keys = [self.latent_key, "mesh_gt"]
        self.data.patch_shape = self.task.scene_shape

    @rank_zero_only
    def log_config(self) -> None:
        """Log experiment configuration to files (rank 0 only in distributed training).

        Serializes the complete configuration to both YAML format (for human readability
        and programmatic loading) and a bash script (for reproducing the exact command).
        Only executes on rank 0 process in distributed training to avoid file conflicts.

        Creates two files in the log directory:
        1. config.yaml: Complete configuration as YAML
        2. launch.sh: Command-line invocation for reproducing the run

        Notes:
            The @rank_zero_only decorator ensures this only runs on the main process
            in distributed training, preventing race conditions and duplicate writes.
        """
        configs = dataclasses.asdict(self)
        logger.info(f"Train {self.name} configs: {configs}")
        with open(os.path.join(self.log_dir, f"config.yaml"), "w") as f:
            yaml.dump(dataclasses.asdict(self), f)

        with open(os.path.join(self.log_dir, "launch.sh"), "w") as f:
            f.write(" ".join(sys.argv))

    def update_model_config(self) -> None:
        """Update model-specific configuration.

        Base implementation is empty. Subclasses should override this method
        to configure model-specific parameters based on other config values.

        Examples of model-specific updates in subclasses:
        - Setting checkpoint paths
        - Configuring model channels based on VAE settings
        - Loading pretrained model configurations
        """
        pass

    def update_and_log_configs(self, out_dir: str) -> None:
        """Execute all configuration updates and log the final configuration.

        Orchestrates the complete configuration setup process by calling all
        update methods in the correct order, then logs the configuration if
        this is a training run.

        The update sequence ensures dependencies are resolved correctly:
        1. Model-specific configuration
        2. Dataset configuration (may depend on model settings)
        3. Experiment directory setup (resume or new)
        4. SLURM job configuration (depends on directories and timestamp)
        5. Configuration logging (only for training tasks)

        Args:
            out_dir: Base output directory for the experiment. Passed to
                resume_or_from_sratch for directory setup.

        Notes:
            Configuration is only logged to files if self.task.name == "train",
            preventing unnecessary file writes during validation or testing.
        """
        self.update_model_config()
        self.update_data_config()
        self.resume_or_from_sratch(out_dir)
        self.update_slurm_config()
        if "train" in self.task.name:
            self.log_config()


@dataclasses.dataclass(kw_only=True)
class VAE(Base):
    """Configuration for sparse variational autoencoder training.

    Extends Base with autoencoder-specific parameters including the model
    architecture, task type, and training hyperparameters. Used for training
    VAEs on sparse 3D voxel data for scene representation learning.

    Attributes:
        model: Sparse VAE model configuration defining the encoder/decoder architecture.
        task: Task specification, one of TrainVAE (training)
            or Reconstruction (inference). Default is training.
        name: Display name for this configuration type.
        target: Python import path to the Lightning module class implementing
            the autoencoder training logic.
        bn_momentum: Momentum parameter for batch normalization layers. Controls
            the exponential moving average of batch statistics.
    """

    model: SparseVAE
    task: Union[TrainVAE, Reconstruction] = field(default_factory=TrainVAE)
    name: str = "VAE"
    target: str = "seen2scene.models.trainer.train_vae.Net"
    bn_momentum: float = 0.1

    def update_model_config(self) -> None:
        """Update autoencoder model configuration with paths and data-dependent parameters.

        Constructs the checkpoint path if specified, formatting it with the voxel size,
        and synchronizes the model's known_ratio with the dataset configuration.

        The checkpoint path follows the structure:
        {log_root}/auto_encoder/{ckpt_path}/checkpoint/vxl_{voxel_size}_last.ckpt

        Notes:
            Modifies the following attributes:
            - self.ckpt_path: Converted to full path with voxel size if not None
            - self.model.known_ratio: Set to match self.data.known_ratio
        """
        voxel_size = float2str(self.data.voxel_size)
        if self.ckpt_path is not None:
            self.ckpt_path = os.path.join(
                self.log_root,
                "auto_encoder",
                self.ckpt_path,
                "checkpoint",
                f"vxl_{voxel_size}_last.ckpt",
            )
        self.model.known_ratio = self.data.known_ratio

    def __post_init__(self) -> None:
        """Initialize autoencoder configuration after dataclass construction.

        Automatically called after the dataclass __init__. Sets up the output
        directory structure and executes all configuration updates.

        The output directory is placed under {log_root}/auto_encoder/, keeping
        autoencoder experiments organized separately from other model types.
        """
        out_dir = os.path.join(self.log_root, "auto_encoder")
        self.update_and_log_configs(out_dir)


@dataclasses.dataclass(kw_only=True)
class Generator(Base):
    """Configuration for flow matching generator training.

    Extends Base with generator-specific parameters for training flow matching
    models on latent representations from a pretrained VAE. The generator learns
    to denoise and generate scene completions in the latent space.

    Attributes:
        model: Flow matching model configuration defining the denoising architecture.
        task: Task specification, one of TrainGen (training)
            Generation (inference), or LargeScaleGeneration (large-scale inference).
            Default is training.
        name: Display name for this configuration type.
        target: Python import path to the Lightning module class implementing
            the generator training logic.
        progressive_end: Training step at which to end progressive training.
            -1 disables progressive training.
        sample_posterior: If True, sample from VAE posterior during training.
            If False, use deterministic encoding (mean only).
        ae_log: Name of the autoencoder experiment folder to load VAE from.
            Used to construct paths to VAE config and checkpoint.
        avg_over: Strategy for averaging metrics over the output patch. Options:
            "region" averages over all voxels in a region, "voxel" does not average and returns per-voxel metrics
        vae_cfg: VAE model configuration loaded from the autoencoder's config.yaml.
            Automatically populated, suppressed from tyro CLI.
        vae_ckpt: Path to VAE checkpoint file. Automatically constructed from ae_log.
        norm_by: Normalization strategy for latent codes. Options: "minmax" scales
            to [0,1], "std" standardizes to mean=0, std=1.
    """

    model: FM
    task: Union[
        TrainGen, Generation, TextToScene, LargeScaleGeneration, LargeScaleTextToScene
    ] = field(default_factory=TrainGen)
    name: str = "Generator"
    target: str = "seen2scene.models.trainer.train_gen.Net"
    progressive_end: int = -1
    sample_posterior: bool = True
    ae_log: Optional[str] = None
    avg_over: Literal["region", "voxel"] = "region"

    vae_cfg: Annotated[Optional[Dict[str, Any]], tyro.conf.Suppress] = None
    vae_ckpt: Optional[str] = None
    norm_by: Literal["minmax", "std"] = "std"

    def resume_ae_from_yaml(self) -> None:
        """Load VAE configuration and checkpoint path from autoencoder experiment.

        Reads the config.yaml from the specified autoencoder experiment directory,
        extracts the model configuration, and constructs the path to the appropriate
        VAE checkpoint based on the current voxel size.

        The VAE checkpoint path follows the structure:
        {log_root}/auto_encoder/{ae_log}/checkpoint/vxl_{voxel_size}_last.ckpt

        Raises:
            FileNotFoundError: If the autoencoder config.yaml doesn't exist.
            KeyError: If the config.yaml doesn't contain a "model" key.

        Notes:
            Modifies the following attributes:
            - self.vae_cfg: Set to the model configuration from the AE experiment
            - self.vae_ckpt: Set to the path of the VAE checkpoint file
        """
        ae_dir = os.path.join(self.log_root, "auto_encoder", self.ae_log)
        with open(os.path.join(ae_dir, "config.yaml"), "r") as yaml_file:
            model_cfg = yaml.unsafe_load(yaml_file)["model"]
            self.vae_cfg = model_cfg
            vxl_size = float2str(self.data.voxel_size)
            self.vae_ckpt = os.path.join(
                ae_dir, "checkpoint", f"vxl_{vxl_size}_last.ckpt"
            )
            logger.info(f"Load VAE config from {ae_dir}")

    def update_data_config(self) -> None:
        """Update dataset configuration based on task requirements.

        Synchronizes data loading keys across train/val/test splits and sets
        the patch shape to match the task's scene shape. This ensures consistent
        data handling across all dataset splits.

        Notes:
            Modifies the following attributes:
            - self.train_data_keys: Set to [self.latent_key, "mesh_gt"]
            - self.val_data_keys: Set to [self.latent_key, "mesh_gt"]
            - self.test_data_keys: Set to [self.latent_key, "mesh_gt"]
            - self.data.patch_shape: Set to self.task.scene_shape
        """
        self.train_data_keys = [self.latent_key]
        self.val_data_keys = self.test_data_keys = [self.latent_key, "mesh_gt"]
        self.data.patch_shape = self.task.scene_shape
        self.data.max_scene_volume = getattr(self.task, "max_scene_volume", None)

        # Append _centered to patch-level task names when objects are centered
        _PATCH_TASKS = {"patch_generation", "patch_completion", "text2scene"}
        if self.data.bbox_rand_shift == 0.0 and self.task.name in _PATCH_TASKS:
            self.task.name = f"{self.task.name}_centered"

        if self.data.csv_path is not None:
            self.task.name = (
                f"{self.task.name}_{os.path.basename(self.data.csv_path).split('.')[0]}"
            )
        else:
            self.task.name = f"{self.task.name}_" + "_".join(
                map(str, self.task.scene_shape)
            )

        if self.task.postfix:
            self.task.name = f"{self.task.name}_{self.task.postfix}"

    def update_model_config(self) -> None:
        """Update generator model configuration with VAE-dependent parameters.

        Configures the generator's input/output channels to match the VAE latent
        space, constructs checkpoint paths, and adjusts architecture based on the
        conditioning strategy

        The checkpoint path follows the structure:
        {log_root}/auto_encoder/{ae_log}/generator/{ckpt_path}/checkpoint/vxl_{voxel_size}_last.ckpt

        Notes:
            Modifies the following attributes:
            - self.ckpt_path: Converted to full path if not None
            - self.model.model_cfg.in_channels: Set based on VAE channels and conditioning
            - self.model.model_cfg.out_channels: Set to VAE channels
        """
        if self.ckpt_path is not None:
            vxl_size = float2str(self.data.voxel_size)
            self.ckpt_path = os.path.join(
                self.log_root,
                "auto_encoder",
                self.ae_log,
                "generator",
                self.ckpt_path,
                "checkpoint",
                f"vxl_{vxl_size}_last.ckpt",
            )

        self.model.model_cfg.in_channels = self.model.model_cfg.out_channels = (
            self.vae_cfg["channels"]
        )

    def __post_init__(self) -> None:
        """Initialize generator configuration after dataclass construction.

        Automatically called after the dataclass __init__. Loads the VAE configuration,
        sets up the output directory structure, and executes all configuration updates.

        The output directory is nested under the autoencoder experiment:
        {log_root}/auto_encoder/{ae_log}/generator/

        This organization keeps generator experiments associated with their VAE.
        """
        out_dir = os.path.join(self.log_root, "auto_encoder", self.ae_log, "generator")
        self.resume_ae_from_yaml()
        self.update_and_log_configs(out_dir)


@dataclasses.dataclass(kw_only=True)
class Control(Generator):
    """Configuration for ControlNet-based conditional generation.

    Extends Generator to add ControlNet-specific parameters for conditional
    scene completion. ControlNet adds spatial control to the generation process
    by conditioning on source inputs like partial observations or semantic maps.

    Attributes:
        model: ControlNet model configuration defining the control network architecture.
        src_key: Data key for the source/control input (e.g., partial scene, depth map).
            Used to load conditioning inputs from the dataset.
        name: Display name for this configuration type.
        gen_log: Name of the generator experiment folder to load base generator from.
            Used to construct paths to generator config and checkpoint.
    """

    model: ControlNet
    task: Union[TrainControl, ValCompletion, Completion, LargeScaleCompletion] = field(
        default_factory=TrainControl
    )
    src_key: Optional[str]
    name: str = "control"
    gen_log: Optional[str] = None

    def update_model_config(self) -> None:
        """Update ControlNet model configuration by loading base generator config.

        Reads the generator's config.yaml to extract the base model configuration,
        which the ControlNet will condition. Also constructs the path to the
        generator checkpoint that will be used as the frozen base model.

        The generator checkpoint path follows the structure:
        {log_root}/auto_encoder/{ae_log}/generator/{gen_log}/checkpoint/vxl_{voxel_size}_last.ckpt

        Raises:
            FileNotFoundError: If the generator config.yaml doesn't exist.
            KeyError: If the config.yaml doesn't contain a "model" key.

        Notes:
            Modifies the following attributes:
            - self.model.gen_cfg: Set to the generator model configuration
            - self.model.gen_ckpt: Set to the path of the generator checkpoint
        """
        gen_dir = os.path.join(
            self.log_root, "auto_encoder", self.ae_log, "generator", self.gen_log
        )
        with open(os.path.join(gen_dir, "config.yaml"), "r") as yaml_file:
            model_cfg = yaml.unsafe_load(yaml_file)["model"]
            self.model.gen_cfg = model_cfg
            vxl_size = float2str(self.data.voxel_size)
            self.model.gen_ckpt = os.path.join(
                gen_dir, "checkpoint", f"vxl_{vxl_size}_last.ckpt"
            )
            logger.info(f"Load generator config from {gen_dir}")

        if self.ckpt_path is not None:
            vxl_size = float2str(self.data.voxel_size)
            self.ckpt_path = os.path.join(
                gen_dir,
                "control",
                self.ckpt_path,
                "checkpoint",
                f"vxl_{vxl_size}_last.ckpt",
            )

    def update_data_config(self) -> None:
        """Update dataset configuration for conditional generation.

        Configures data loading to include both source/control inputs (src_key)
        and target latents (latent_key) for training the ControlNet. Both inputs
        are needed: the source for conditioning and the latent for the target.

        Notes:
            Modifies the following attributes:
            - self.train_data_keys: Set to [src_key, latent_key]
            - self.val_data_keys: Set to [src_key, latent_key]
            - self.test_data_keys: Set to [src_key, latent_key]
            - self.data.patch_shape: Set to self.task.scene_shape
        """
        self.train_data_keys = [self.src_key, self.latent_key]
        self.val_data_keys = self.test_data_keys = [
            self.src_key,
            self.latent_key,
            "mesh_gt",
        ]
        self.data.patch_shape = self.task.scene_shape
        self.data.max_scene_volume = getattr(self.task, "max_scene_volume", None)

        # Append _centered to patch-level task names when objects are centered
        _PATCH_TASKS = {"patch_generation", "patch_completion", "text2scene"}
        if self.data.bbox_rand_shift == 0.0 and self.task.name in _PATCH_TASKS:
            self.task.name = f"{self.task.name}_centered"

        if self.data.csv_path is not None:
            self.task.name = (
                f"{self.task.name}_{os.path.basename(self.data.csv_path).split('.')[0]}"
            )
        if self.data.patch_names is not None:
            self.task.name = f"{self.task.name}_spec_patches"
        if getattr(self.task, "drop_bbox", False):
            self.task.name = f"{self.task.name}_drop_bbox"
        if self.task.postfix:
            self.task.name = f"{self.task.name}_{self.task.postfix}"

    def __post_init__(self) -> None:
        """Initialize ControlNet configuration after dataclass construction.

        Automatically called after the dataclass __init__. Loads both VAE and
        generator configurations, sets up the output directory structure, and
        executes all configuration updates.

        The output directory is nested under the generator experiment:
        {log_root}/auto_encoder/{ae_log}/generator/{gen_log}/control/

        This organization keeps ControlNet experiments associated with their
        base generator and VAE.
        """
        out_dir = os.path.join(
            self.log_root,
            "auto_encoder",
            self.ae_log,
            "generator",
            self.gen_log,
            "control",
        )
        self.resume_ae_from_yaml()
        self.update_and_log_configs(out_dir)
