import dataclasses
from typing import Optional


@dataclasses.dataclass(kw_only=True)
class SparseVAE:
    """Configuration for sparse variational autoencoder using fVDB representation.

    Defines the architecture and training parameters for a VAE that operates on
    sparse voxel grids using the fVDB (Fast Voxel Data Base) data structure.
    The VAE compresses 3D scenes into compact latent representations for efficient
    generation and manipulation.

    Attributes:
        name: Display name for the model architecture.
        target: Python import path to the SparseVAE implementation class.
        encoder: Python import path to the encoder network class.
        decoder: Python import path to the decoder network class.
        channels: Number of channels in the latent representation. Smaller values
            (e.g., 8) provide more compression but may lose detail.
        factor: Spatial downsampling factor from input to latent space. Factor of 8
            means a 256³ input becomes 32³ in latent space.
        is_add_dec: If True, use additive decoder skip connections. If False, use
            concatenative skip connections.
        f_maps: Base number of feature channels in encoder/decoder. Deeper layers
            multiply this (e.g., 64, 128, 256). Higher values increase capacity.
        order: Operation order in residual blocks. "gcs" means GroupNorm → Conv → SiLU.
        num_res_blocks: Number of residual blocks per resolution level. More blocks
            increase capacity but slow training.
        num_groups: Number of groups for Group Normalization. 32 is standard for
            good normalization without excessive overhead.
        use_attention: If True, add self-attention layers at lower resolutions.
            Helps capture global structure but increases compute.
        use_residual: If True, use residual connections in encoder/decoder blocks.
            Improves gradient flow and training stability.
        use_checkpoint: If True, use gradient checkpointing to reduce memory usage
            at the cost of increased computation during backward pass.
        unstable_cutoff_threshold: Threshold for detecting unstable voxel regions.
            Voxels with gradients above 0.15 are considered unstable.
        max_down_time: Maximum number of downsampling operations. 6 downsamples
            reduces 256³ to 4³ spatial resolution.
        with_color_branch: If True, add a separate decoder branch for RGB color
            prediction in addition to geometry.
        with_semantic_branch: If True, add a separate decoder branch for semantic
            label prediction.
        gaussian_tau: Tau parameter for Gaussian distance weighting in loss computation.
            Formula: exp(-u² / (2 * tau²)). Smaller tau (e.g., 0.5) concentrates
            weight near zero distance. None disables Gaussian weighting.
        feed_gt_structure: If True, feed ground truth voxel structure to decoder
            during training. Used for ablation studies of structure learning.
        double_z: If True, use double latent channels for separate mean and log-variance
            in VAE. Standard practice for reparameterization trick.
        kl_weight: Weight for KL divergence loss in VAE objective. 1.0 is standard
            β-VAE weighting. Lower values create more flexible latent space.
        structure_weight: Weight for structure prediction loss. 1e2 emphasizes learning
            correct occupancy structure.
        geometry_weight: Weight for geometry (TSDF) prediction loss. 1e2 emphasizes
            learning accurate surface distances.

    Notes:
        The architecture uses sparse convolutions on fVDB grids, processing only
        occupied voxels for computational efficiency on 3D scenes.

    Examples:
        Standard configuration for 8-channel latent with 8x downsampling:
        >>> config = SparseVAE(channels=8, factor=8, f_maps=64)

        Higher capacity model with attention:
        >>> config = SparseVAE(channels=16, f_maps=128, use_attention=True)
    """
    name: str = "fVDB VAE"
    target: str = "seen2scene.models.fvdb.sparse_vae.SparseVAE"
    encoder: str = "seen2scene.models.fvdb.encoder.Encoder"
    decoder: str = "seen2scene.models.fvdb.decoder.Decoder"

    channels: int = 8
    factor: int = 8
    is_add_dec: bool = False
    f_maps: int = 64  # 32, 64, 128
    order: str = "gcs"
    num_res_blocks: int = 1
    num_groups: int = 32
    use_attention: bool = False
    use_residual: bool = True
    use_checkpoint: bool = True

    unstable_cutoff_threshold: float = 0.15
    max_down_time: int = 6  # From 256 to 4

    with_color_branch: bool = False
    with_semantic_branch: bool = False

    # exp(-u^2 / (2 * tau^2)), in (0, 1), controls how quickly weight falls off, Smaller tau, stronger weight near 0.0
    gaussian_tau: Optional[float] = None  # 0.5
    feed_gt_structure: bool = False
    double_z: bool = True

    kl_weight: float = 1.0
    structure_weight: float = 1.0e2
    geometry_weight: float = 1.0e2

