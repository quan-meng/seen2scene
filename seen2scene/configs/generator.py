import dataclasses
from enum import IntEnum
from typing import *
import tyro


class Category(IntEnum):
    """Enumeration of semantic category types for conditioning.

    Defines special category tokens used in semantic conditioning for generation.

    Attributes:
        END: End-of-sequence token (value 1) marking the end of a category list.
        OTHER: Catch-all category (value 2) for objects not in the main category set.
    """

    END = 1
    OTHER = 2


@dataclasses.dataclass(kw_only=True)
class Empty:
    """Empty configuration placeholder for unconditional generation.

    Used when no conditioning is applied to the generator. Provides a consistent
    interface even when conditioning configuration is not needed.

    Attributes:
        name: Identifier for empty/unconditional configuration.
    """

    name: str = "Empty"


@dataclasses.dataclass(kw_only=True)
class FlowMatchEuler:
    """Configuration for Flow Matching with Euler discrete time stepping.

    Defines the scheduler for flow matching models, which learn to transform
    noise to data through continuous-time ordinary differential equations (ODEs).
    Uses Euler method for discrete time integration.

    Attributes:
        target: Python import path to the FlowMatchEulerDiscreteScheduler implementation
            from the diffusers library.
        num_train_timesteps: Number of discrete timesteps in the diffusion process
            during training. 1000 provides fine-grained temporal resolution for
            learning smooth flow trajectories.
    """

    target: str = "diffusers.FlowMatchEulerDiscreteScheduler"
    num_train_timesteps: int = 1000


@dataclasses.dataclass(kw_only=True)
class SparseDiT:
    """Configuration for Sparse Diffusion Transformer architecture.

    Defines a Diffusion Transformer (DiT) that operates on sparse voxel representations
    using efficient attention mechanisms. The architecture uses transformer blocks
    with adaptive normalization for timestep and conditioning injection.

    Attributes:
        name: Display name for the model architecture.
        target: Python import path to the SparseDiT implementation class.
        in_channels: Number of input latent channels from the VAE encoder.
        out_channels: Number of output latent channels to match VAE decoder input.
        model_channels: Hidden dimension of transformer blocks. Higher values (e.g., 768)
            increase model capacity. Options: 512, 768, 1024.
        context_dim: Dimension of cross-attention context for conditioning. None means
            no cross-attention conditioning is used.
        num_blocks: Number of transformer blocks in the model. 28 blocks creates a
            deep model for complex scene understanding.
        num_heads: Number of attention heads in multi-head self-attention. 16 heads
            allows learning diverse attention patterns.
        num_head_channels: Dimension per attention head. Total attention dimension is
            num_heads * num_head_channels. 32 channels per head is standard.
        num_kv_heads: Number of key-value heads for grouped-query attention. 2 heads
            reduces memory while maintaining quality.
        mlp_ratio: Ratio of MLP hidden dimension to model_channels. 4.0 means the
            feed-forward network expands to 4x the transformer dimension.
        factor: Scaling factor for model initialization or feature dimensions.
        use_checkpoint: If True, use gradient checkpointing to trade computation for
            memory. Helpful when using flash-attention with limited GPU memory.
        share_mod: If True, share modulation parameters across transformer blocks.
            Reduces parameters but may limit expressiveness.
        qk_rms_norm: If True, apply RMS normalization to query and key before attention
            in self-attention layers. Improves training stability.
        qk_rms_norm_cross: If True, apply RMS normalization to query and key in
            cross-attention layers for conditioning.
        attn_mode: Attention computation mode. "full" computes all-to-all attention,
            "windowed" restricts to local windows, "serialized" processes sequentially.
        zero_out: If True, initialize output projection to zero. Common in diffusion
            models for stable training start.
        use_ssa: If True, use Sparse Structured Attention for efficient computation
            on sparse voxel grids.
        window_size: Spatial window size for windowed or sparse attention. 4 means
            4x4x4 voxel windows in 3D.
        use_shift: If True, use shifted windows in attention for better coverage
            (similar to Swin Transformer).
        selection_block_size: Block size for selecting important voxels in sparse
            attention. Controls granularity of sparsity.
        compression_block_size: Block size for compressing sparse representations.
            Smaller values (e.g., 2) provide finer compression control.
        topk: Number of top-k voxels to select per block in sparse attention.
            8 voxels per block balances efficiency and quality.
        compression_version: Version identifier for compression algorithm. "v2"
            indicates the latest compression implementation.

    Notes:
        The SparseDiT is designed for 3D scene generation, processing only occupied
        voxels through sparse operations for computational efficiency.

    Examples:
        Standard configuration for high-quality generation:
        >>> config = SparseDiT(model_channels=768, num_blocks=28, num_heads=16)

        Lightweight configuration for faster inference:
        >>> config = SparseDiT(model_channels=512, num_blocks=12, num_heads=8)
    """

    name: str = "SparseDiT"
    target: str = "seen2scene.models.sparse.dit.SparseDiT"
    in_channels: int = 8
    out_channels: int = 8
    model_channels: int = 768  # 1024, 768, 512
    context_dim: Optional[int] = None
    num_blocks: int = 28
    num_heads: int = 16
    num_head_channels: int = 32
    num_kv_heads: int = 2
    mlp_ratio: float = 4
    pe_mode: Literal["ape", "rope"] = "rope"
    factor: float = 1.0
    use_checkpoint: bool = True  # Helps for flash-attention
    share_mod: bool = False
    qk_rms_norm: bool = True
    qk_rms_norm_cross: bool = True
    attn_mode: Literal["full", "windowed", "serialized"] = "full"
    zero_out: bool = True

    use_ssa: bool = False
    window_size: int = 4
    use_shift: bool = True
    selection_block_size: int = 4
    compression_block_size: int = 2
    topk: int = 8
    compression_version: str = "v2"


@dataclasses.dataclass(kw_only=True)
class FM:
    """Configuration for Flow Matching generative model.

    Defines parameters for training and inference with flow matching, a continuous
    normalizing flow approach to generative modeling. Combines a SparseDiT backbone
    with a flow matching scheduler and various training techniques.

    Attributes:
        model_cfg: Configuration for the SparseDiT backbone network.
        scheduler: Configuration for the flow matching time scheduler.
        name: Display name for the model type.
        target: Python import path to the flow matching Model implementation.
        cond_drop_prob: Probability of dropping conditioning during training for
            classifier-free guidance. 0.0 means always conditioned, 0.1 enables
            guidance by randomly training without conditioning 10% of the time.
        use_ema: If True, maintain exponential moving average of model weights.
            EMA weights typically produce better sample quality.
        num_inference_steps: Number of integration steps during sampling. 50 steps
            balances quality and speed. More steps improve quality but slow inference.
        weighting_scheme: Loss weighting scheme across timesteps. "logit_normal"
            uses logit-normal distribution to emphasize certain timesteps.
        logit_mean: Mean parameter for logit-normal weighting distribution. 0.0
            centers the distribution.
        logit_std: Standard deviation for logit-normal weighting. 1.0 provides
            moderate variance in timestep weighting.
        mode_scale: Scale parameter for mode-based weighting scheme. Only used
            when weighting_scheme is "mode". 1.29 is the standard value.
        precondition_outputs: If True, precondition model outputs as in EDM
            (Elucidating Diffusion Models). Affects target calculation in loss.
            False produces smoother loss curves during training.

    Notes:
        Flow matching learns continuous-time dynamics from noise to data, providing
        an alternative to discrete diffusion processes with theoretical advantages.

    Examples:
        Standard configuration with classifier-free guidance:
        >>> config = FM(
        ...     model_cfg=SparseDiT(),
        ...     scheduler=FlowMatchEuler(),
        ...     cond_drop_prob=0.1,
        ...     num_inference_steps=50
        ... )
    """

    model_cfg: SparseDiT
    scheduler: FlowMatchEuler

    name: str = "FM"
    target: str = "seen2scene.models.backbone.flow_matching.Model"
    cond_drop_prob: float = 0.1
    use_ema: bool = True
    num_inference_steps: int = 50
    weighting_scheme: str = "logit_normal"
    # mean to use when using the `'logit_normal'` weighting scheme.
    logit_mean: float = 0.0
    # std to use when using the `'logit_normal'` weighting scheme.
    logit_std: float = 1.0
    # Scale of mode weighting scheme. Only effective when using the `'mode'` as the `weighting_scheme`.
    mode_scale: float = 1.29
    # Flag indicating if we are preconditioning the model outputs or not as done in EDM. This affects how model `target` is calculated.
    precondition_outputs: bool = False  # False: loss curve more smooth
    # The number of denoising steps. More denoising steps usually lead to a higher quality image at the expense of slower inference.
    band_weight: float = 0.5
    zero_noise: bool = False
    use_distance_rope: bool = False
    use_label_augmentation: bool = False


@dataclasses.dataclass(kw_only=True)
class ControlNet:
    """Configuration for ControlNet conditional generation.

    A trainable branch (SparseDiT clone) processes spatial conditions and injects
    per-layer control signals into a frozen base generator. The branch is initialized
    from pretrained base weights with zero-initialized input/output layers.

    Attributes:
        gen_cfg: Generator model configuration loaded from the base generator's
            saved config. Automatically populated, suppressed from CLI arguments.
        gen_ckpt: Path to the base generator checkpoint file. The ControlNet will
            use this frozen generator as its backbone.
        name: Display name for the model type.
        target: Python import path to the ControlNet implementation class.
    """

    gen_cfg: Annotated[Optional[Dict[str, Any]], tyro.conf.Suppress] = None
    gen_ckpt: Optional[str] = None
    zero_proj_rank: int = 64
    """Rank of low-rank zero projections in the control branch."""

    name: str = "ControlNet"
    target: str = "seen2scene.models.backbone.controlnet.Net"
