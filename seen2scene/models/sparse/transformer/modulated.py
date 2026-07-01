from typing import *
import torch
import torch.nn as nn
from ..basic import SparseTensor
from ..attention import SparseMultiHeadAttention, SerializeMode, SpatialSparseAttention
from .blocks import LayerNorm32, SparseFeedForwardNet


class ModulatedSparseTransformerBlock(nn.Module):
    """
    Sparse Transformer block (MSA + FFN) with adaptive layer norm conditioning.
    """

    def __init__(
        self,
        channels: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        attn_mode: Literal[
            "full", "shift_window", "shift_sequence", "shift_order", "swin"
        ] = "full",
        window_size: Optional[int] = None,
        shift_sequence: Optional[int] = None,
        shift_window: Optional[Tuple[int, int, int]] = None,
        serialize_mode: Optional[SerializeMode] = None,
        use_checkpoint: bool = False,
        use_rope: bool = False,
        qk_rms_norm: bool = False,
        qkv_bias: bool = True,
        share_mod: bool = False,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.share_mod = share_mod
        self.norm1 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.norm2 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.attn = SparseMultiHeadAttention(
            channels,
            num_heads=num_heads,
            attn_mode=attn_mode,
            window_size=window_size,
            shift_sequence=shift_sequence,
            shift_window=shift_window,
            serialize_mode=serialize_mode,
            qkv_bias=qkv_bias,
            use_rope=use_rope,
            qk_rms_norm=qk_rms_norm,
        )
        self.mlp = SparseFeedForwardNet(
            channels,
            mlp_ratio=mlp_ratio,
        )
        if not share_mod:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(), nn.Linear(channels, 6 * channels, bias=True)
            )

    def _forward(self, x: SparseTensor, mod: torch.Tensor) -> SparseTensor:
        if self.share_mod:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod.chunk(
                6, dim=1
            )
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                self.adaLN_modulation(mod).chunk(6, dim=1)
            )
        h = x.replace(self.norm1(x.feats))
        h = h * (1 + scale_msa) + shift_msa
        h = self.attn(h)
        h = h * gate_msa
        x = x + h
        h = x.replace(self.norm2(x.feats))
        h = h * (1 + scale_mlp) + shift_mlp
        h = self.mlp(h)
        h = h * gate_mlp
        x = x + h
        return x

    def forward(self, x: SparseTensor, mod: torch.Tensor) -> SparseTensor:
        if self.use_checkpoint and self.training:
            return torch.utils.checkpoint.checkpoint(
                self._forward, x, mod, use_reentrant=False
            )
        else:
            return self._forward(x, mod)


class ModulatedSparseTransformerCrossBlock(nn.Module):
    """
    Sparse Transformer cross-attention block (MSA + MCA + FFN) with adaptive layer norm conditioning.
    """

    def __init__(
        self,
        channels: int,
        num_heads: int,
        ctx_channels: Optional[int] = None,
        mlp_ratio: float = 4.0,
        attn_mode: Literal[
            "full", "shift_window", "shift_sequence", "shift_order", "swin"
        ] = "full",
        window_size: Optional[int] = None,
        shift_sequence: Optional[int] = None,
        shift_window: Optional[Tuple[int, int, int]] = None,
        serialize_mode: Optional[SerializeMode] = None,
        use_checkpoint: bool = False,
        compression_version: str = "v2",
        use_rope: bool = False,
        qk_rms_norm: bool = False,
        qk_rms_norm_cross: bool = False,
        qkv_bias: bool = True,
        share_mod: bool = False,
        use_ssa: bool = True,
        num_kv_heads: int = 2,
        compression_block_size: int = 4,
        selection_block_size: int = 8,
        topk: int = 8,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.ctx_channels = ctx_channels

        self.share_mod = share_mod
        self.norm1 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.norm2 = LayerNorm32(channels, elementwise_affine=True, eps=1e-6)
        if use_ssa:
            self.self_attn = SpatialSparseAttention(
                channels,
                num_q_heads=num_heads,
                num_kv_heads=num_kv_heads,
                head_dim=channels // num_heads,
                compression_block_size=compression_block_size,
                compression_version=compression_version,
                selection_block_size=selection_block_size,
                topk=topk,
                window_size=window_size,
                shift_window=shift_window,
            )
        else:
            self.self_attn = SparseMultiHeadAttention(
                channels,
                num_heads=num_heads,
                type="self",
                attn_mode=attn_mode,
                window_size=window_size,
                shift_sequence=shift_sequence,
                shift_window=shift_window,
                serialize_mode=serialize_mode,
                qkv_bias=qkv_bias,
                use_rope=use_rope,
                qk_rms_norm=qk_rms_norm,
            )

        if ctx_channels is not None:
            self.norm3 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
            self.cross_attn = SparseMultiHeadAttention(
                channels,
                ctx_channels=ctx_channels,
                num_heads=num_heads,
                type="cross",
                attn_mode="full",
                qkv_bias=qkv_bias,
                qk_rms_norm=qk_rms_norm_cross,
            )

        self.mlp = SparseFeedForwardNet(
            channels,
            mlp_ratio=mlp_ratio,
        )
        if not share_mod:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(), nn.Linear(channels, 6 * channels, bias=True)
            )

    def _forward(
        self,
        x: SparseTensor,
        mod: torch.Tensor,
        context: SparseTensor,
        causal: bool = False,
    ) -> SparseTensor:
        if self.share_mod:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod.chunk(
                6, dim=1
            )
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                self.adaLN_modulation(mod).chunk(6, dim=1)
            )
        h = x.replace(self.norm1(x.feats))

        feats_h = h.feats
        layouts = h.layout
        ada_r1 = []
        for i in range(len(layouts)):
            ada_r1.append(
                feats_h[layouts[i]] * (1 + scale_msa[i : i + 1]) + shift_msa[i : i + 1]
            )
        h = h.replace(torch.cat(ada_r1, dim=0))
        h = self.self_attn(h, causal=causal)

        feats_h = h.feats
        layouts = h.layout
        ada_r2 = []
        for i in range(len(layouts)):
            ada_r2.append(feats_h[layouts[i]] * gate_msa[i : i + 1])
        h = h.replace(torch.cat(ada_r2, dim=0))

        x = x + h
        h = x.replace(self.norm2(x.feats))

        if self.ctx_channels is not None:
            h = self.cross_attn(h, context)
            x = x + h
            h = x.replace(self.norm3(x.feats))

        feats_h = h.feats
        layouts = h.layout
        ada_r3 = []
        for i in range(len(layouts)):
            ada_r3.append(
                feats_h[layouts[i]] * (1 + scale_mlp[i : i + 1]) + shift_mlp[i : i + 1]
            )
        h = h.replace(torch.cat(ada_r3, dim=0))
        h = self.mlp(h)

        feats_h = h.feats
        layouts = h.layout
        ada_r4 = []
        for i in range(len(layouts)):
            ada_r4.append(feats_h[layouts[i]] * gate_mlp[i : i + 1])
        h = h.replace(torch.cat(ada_r4, dim=0))

        x = x + h
        return x

    def forward(
        self,
        x: SparseTensor,
        mod: torch.Tensor,
        context: SparseTensor,
        causal: bool = False,
    ) -> SparseTensor:
        if self.use_checkpoint and self.training:
            return torch.utils.checkpoint.checkpoint(
                self._forward, x, mod, context, causal=causal, use_reentrant=False
            )
        else:
            return self._forward(x, mod, context, causal=causal)
