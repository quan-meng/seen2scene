from typing import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from .. import SparseTensor
from .full_attn import sparse_scaled_dot_product_attention
from .serialized_attn import (
    SerializeMode,
    sparse_serialized_scaled_dot_product_self_attention,
)
from .windowed_attn import sparse_windowed_scaled_dot_product_self_attention


class RotaryPositionEmbedder(nn.Module):
    def __init__(self, hidden_size: int, in_channels: int = 3):
        super().__init__()
        assert hidden_size % 2 == 0, "Hidden size must be divisible by 2"
        self.hidden_size = hidden_size
        self.in_channels = in_channels
        self.freq_dim = hidden_size // in_channels // 2
        self.freqs = torch.arange(self.freq_dim, dtype=torch.float32) / self.freq_dim
        self.freqs = 1.0 / (10000**self.freqs)

    def _get_phases(self, indices: torch.Tensor) -> torch.Tensor:
        self.freqs = self.freqs.to(indices.device)
        phases = torch.outer(indices, self.freqs)
        phases = torch.polar(torch.ones_like(phases), phases)
        return phases

    # TODO: still need to check
    def _rotary_embedding(self, x: torch.Tensor, phases: torch.Tensor) -> torch.Tensor:
        x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        x_rotated = x_complex * phases[:, None]
        x_embed = (
            torch.view_as_real(x_rotated).reshape(*x_rotated.shape[:-1], -1).to(x.dtype)
        )
        return x_embed

    def forward(
        self, x: torch.Tensor, indices: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x (sp.SparseTensor): [..., N, H, D] tensor of queries/keys
            indices (torch.Tensor): [..., N, C] tensor of spatial positions
        """
        if indices is None:
            indices = torch.arange(x.shape[-2], device=x.device)
            if len(x.shape) > 2:
                indices = indices.unsqueeze(0).expand(x.shape[:-2] + (-1,))

        phases = self._get_phases(indices.reshape(-1)).reshape(*indices.shape[:-1], -1)

        if phases.shape[1] < self.hidden_size // 2:
            phases = torch.cat(
                [
                    phases,
                    torch.polar(
                        torch.ones(
                            *phases.shape[:-1],
                            self.hidden_size // 2 - phases.shape[1],
                            device=phases.device,
                        ),
                        torch.zeros(
                            *phases.shape[:-1],
                            self.hidden_size // 2 - phases.shape[1],
                            device=phases.device,
                        ),
                    ),
                ],
                dim=-1,
            )
        embed = self._rotary_embedding(x, phases)
        return embed


class SparseMultiHeadRMSNorm(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(heads, dim))

    def forward(
        self, x: Union[SparseTensor, torch.Tensor]
    ) -> Union[SparseTensor, torch.Tensor]:
        x_type = x.dtype
        x = x.float()
        if isinstance(x, SparseTensor):
            x = x.replace(F.normalize(x.feats, dim=-1))
        else:
            x = F.normalize(x, dim=-1)
        out = (x * self.gamma * self.scale).to(x_type)

        return out


class SparseMultiHeadAttention(nn.Module):
    def __init__(
        self,
        channels: int,
        num_heads: int,
        ctx_channels: Optional[int] = None,
        type: Literal["self", "cross"] = "self",
        attn_mode: Literal["full", "serialized", "windowed"] = "full",
        window_size: Optional[int] = None,
        shift_sequence: Optional[int] = None,
        shift_window: Optional[Tuple[int, int, int]] = None,
        serialize_mode: Optional[SerializeMode] = None,
        qkv_bias: bool = True,
        use_rope: bool = False,
        qk_rms_norm: bool = False,
    ):
        super().__init__()
        assert channels % num_heads == 0
        assert type in ["self", "cross"], f"Invalid attention type: {type}"
        assert attn_mode in [
            "full",
            "serialized",
            "windowed",
        ], f"Invalid attention mode: {attn_mode}"
        assert (
            type == "self" or attn_mode == "full"
        ), "Cross-attention only supports full attention"
        assert (
            type == "self" or use_rope is False
        ), "Rotary position embeddings only supported for self-attention"
        self.channels = channels
        self.ctx_channels = ctx_channels if ctx_channels is not None else channels
        self.num_heads = num_heads
        self._type = type
        self.attn_mode = attn_mode
        self.window_size = window_size
        self.shift_sequence = shift_sequence
        self.shift_window = shift_window
        self.serialize_mode = serialize_mode
        self.use_rope = use_rope
        self.qk_rms_norm = qk_rms_norm

        if self._type == "self":
            self.to_qkv = nn.Linear(channels, channels * 3, bias=qkv_bias)
        else:
            self.to_q = nn.Linear(channels, channels, bias=qkv_bias)
            self.to_kv = nn.Linear(self.ctx_channels, channels * 2, bias=qkv_bias)

        if self.qk_rms_norm:
            self.q_rms_norm = SparseMultiHeadRMSNorm(channels // num_heads, num_heads)
            self.k_rms_norm = SparseMultiHeadRMSNorm(channels // num_heads, num_heads)

        self.to_out = nn.Linear(channels, channels)

        if use_rope:
            self.rope = RotaryPositionEmbedder(channels // num_heads)

    @staticmethod
    def _linear(
        module: nn.Linear, x: Union[SparseTensor, torch.Tensor]
    ) -> Union[SparseTensor, torch.Tensor]:
        if isinstance(x, SparseTensor):
            return x.replace(module(x.feats))
        else:
            return module(x)

    @staticmethod
    def _reshape_chs(
        x: Union[SparseTensor, torch.Tensor], shape: Tuple[int, ...]
    ) -> Union[SparseTensor, torch.Tensor]:
        if isinstance(x, SparseTensor):
            return x.reshape(*shape)
        else:
            return x.reshape(*x.shape[:2], *shape)

    def _fused_pre(
        self, x: Union[SparseTensor, torch.Tensor], num_fused: int
    ) -> Union[SparseTensor, torch.Tensor]:
        if isinstance(x, SparseTensor):
            x_feats = x.feats.unsqueeze(0)
        else:
            x_feats = x
        x_feats = x_feats.reshape(*x_feats.shape[:2], num_fused, self.num_heads, -1)
        return x.replace(x_feats.squeeze(0)) if isinstance(x, SparseTensor) else x_feats

    def _rope(
        self,
        q: torch.Tensor,
        q_coords: torch.Tensor,
        k: torch.Tensor,
        k_coords: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            q (sp.SparseTensor): [..., N, D] tensor of queries
            k (sp.SparseTensor): [..., N, D] tensor of keys
        """
        q = self.rope(q, q_coords)
        k = self.rope(k, k_coords)

        return q, k

    def forward(
        self,
        x: Union[SparseTensor, torch.Tensor],
        context: Optional[Union[SparseTensor, torch.Tensor]] = None,
        causal: bool = False,
    ) -> Union[SparseTensor, torch.Tensor]:
        if self._type == "self":
            qkv = self._linear(self.to_qkv, x)
            qkv = self._fused_pre(qkv, num_fused=3)
            if self.use_rope:
                q, k, v = qkv.unbind(dim=1)  # [T, H, C]
                q_feats, k_feats = self._rope(
                    q.feats, q.coords[:, 1:], k.feats, k.coords[:, 1:]
                )
                qkv = qkv.replace(torch.stack([q_feats, k_feats, v.feats], dim=1))
            if self.qk_rms_norm:
                q, k, v = qkv.unbind(dim=1)
                q = self.q_rms_norm(q)
                k = self.k_rms_norm(k)
                qkv = qkv.replace(torch.stack([q.feats, k.feats, v.feats], dim=1))
            if self.attn_mode == "full":
                h = sparse_scaled_dot_product_attention(qkv, causal=causal)
            elif self.attn_mode == "serialized":
                h = sparse_serialized_scaled_dot_product_self_attention(
                    qkv,
                    self.window_size,
                    serialize_mode=self.serialize_mode,
                    shift_sequence=self.shift_sequence,
                    shift_window=self.shift_window,
                )
            elif self.attn_mode == "windowed":
                h = sparse_windowed_scaled_dot_product_self_attention(
                    qkv, self.window_size, shift_window=self.shift_window
                )
        else:
            q = self._linear(self.to_q, x)
            q = self._reshape_chs(q, (self.num_heads, -1))
            kv = self._linear(self.to_kv, context)
            kv = self._fused_pre(kv, num_fused=2)
            if self.qk_rms_norm:
                q = self.q_rms_norm(q)
                if isinstance(kv, SparseTensor):
                    k, v = kv.unbind(dim=1)
                    k = self.k_rms_norm(k)
                    kv = kv.replace(
                        torch.stack([k.feats, v.feats], dim=1)
                    )  # [T_KV, 2, H, C]
                else:
                    k, v = kv.unbind(dim=2)
                    k = self.k_rms_norm(k)
                    kv = torch.stack([k, v], dim=2)  # [B, L, 2, H, C]
            h = sparse_scaled_dot_product_attention(q, kv)
        h = self._reshape_chs(h, (-1,))
        h = self._linear(self.to_out, h)
        return h
