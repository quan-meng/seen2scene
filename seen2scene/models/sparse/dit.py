from typing import *
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from .. import sparse as sp
from .transformer import ModulatedSparseTransformerCrossBlock


class AbsolutePositionEmbedder(nn.Module):
    """
    Embeds spatial positions into vector representations.
    """

    def __init__(self, channels: int, in_channels: int = 3):
        super().__init__()
        self.channels = channels
        self.in_channels = in_channels
        self.freq_dim = channels // in_channels // 2
        self.freqs = torch.arange(self.freq_dim, dtype=torch.float32) / self.freq_dim
        self.freqs = 1.0 / (10000**self.freqs)

    def _sin_cos_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Create sinusoidal position embeddings.

        Args:
            x: a 1-D Tensor of N indices

        Returns:
            an (N, D) Tensor of positional embeddings.
        """
        self.freqs = self.freqs.to(x.device)
        out = torch.outer(x, self.freqs)
        out = torch.cat([torch.sin(out), torch.cos(out)], dim=-1)
        return out

    def forward(self, x: torch.Tensor, factor: float = None) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): (N, D) tensor of spatial positions
        """
        if factor is not None:
            x = x * factor
        N, D = x.shape
        assert (
            D == self.in_channels
        ), "Input dimension must match number of input channels"
        embed = self._sin_cos_embedding(x.reshape(-1))
        embed = embed.reshape(N, -1)
        if embed.shape[1] < self.channels:
            embed = torch.cat(
                [
                    embed,
                    torch.zeros(N, self.channels - embed.shape[1], device=embed.device),
                ],
                dim=-1,
            )
        return embed


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.

        Args:
            t: a 1-D Tensor of N indices, one per batch element.
                These may be fractional.
            dim: the dimension of the output.
            max_period: controls the minimum frequency of the embeddings.

        Returns:
            an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -np.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_freq = t_freq.to(self.mlp[0].weight.dtype)
        t_emb = self.mlp(t_freq)
        return t_emb


class SparseDiT(nn.Module):
    def __init__(
        self,
        in_channels: int,
        model_channels: int,
        out_channels: int,
        num_blocks: int,
        context_dim: Optional[int] = None,
        num_heads: Optional[int] = None,
        num_head_channels: Optional[int] = 64,
        num_kv_heads: Optional[int] = 2,
        compression_block_size: int = 1,
        selection_block_size: int = 8,
        topk: int = 8,
        compression_version: str = "v2",
        mlp_ratio: float = 4,
        pe_mode: Literal["ape", "rope"] = "ape",
        use_checkpoint: bool = False,
        share_mod: bool = False,
        qk_rms_norm: bool = False,
        qk_rms_norm_cross: bool = False,
        factor: float = 1.0,
        window_size: Optional[int] = 8,
        use_shift: bool = True,
        use_ssa: bool = False,
        zero_in: bool = False,
        zero_out: bool = False,
        attn_mode: Literal["full", "windowed", "serialized"] = "full",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.context_dim = context_dim
        self.out_channels = out_channels
        self.num_blocks = num_blocks
        self.num_heads = num_heads or model_channels // num_head_channels
        self.mlp_ratio = mlp_ratio
        self.pe_mode = pe_mode
        self.use_checkpoint = use_checkpoint
        self.share_mod = share_mod
        self.qk_rms_norm = qk_rms_norm
        self.qk_rms_norm_cross = qk_rms_norm_cross
        self.factor = factor
        self.attn_mode = attn_mode
        self.compression_block_size = compression_block_size
        self.selection_block_size = selection_block_size
        self.zero_in = zero_in
        self.zero_out = zero_out

        self.t_embedder = TimestepEmbedder(model_channels)
        if share_mod:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(), nn.Linear(model_channels, 6 * model_channels, bias=True)
            )
        if pe_mode == "ape":
            self.pos_embedder = AbsolutePositionEmbedder(model_channels)

        self.input_layer = sp.SparseLinear(in_channels, model_channels)

        self.blocks = nn.ModuleList(
            [
                ModulatedSparseTransformerCrossBlock(
                    model_channels,
                    num_heads=self.num_heads,
                    ctx_channels=context_dim,
                    num_kv_heads=num_kv_heads,
                    compression_block_size=compression_block_size,
                    selection_block_size=selection_block_size,
                    topk=topk,
                    mlp_ratio=self.mlp_ratio,
                    attn_mode=attn_mode,
                    compression_version=compression_version,
                    use_checkpoint=self.use_checkpoint,
                    use_rope=(pe_mode == "rope"),
                    share_mod=self.share_mod,
                    qk_rms_norm=self.qk_rms_norm,
                    qk_rms_norm_cross=self.qk_rms_norm_cross,
                    window_size=window_size,
                    shift_window=(
                        window_size // 2 * (_ % 2) if use_shift else window_size // 2
                    ),
                    use_ssa=use_ssa,
                )
                for _ in range(num_blocks)
            ]
        )

        self.out_layer = sp.SparseLinear(model_channels, out_channels)

        self.initialize_weights()

    @property
    def device(self) -> torch.device:
        """
        Return the device of the model.
        """
        return next(self.parameters()).device

    def initialize_weights(self) -> None:
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear) or isinstance(module, sp.SparseLinear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        if self.share_mod:
            nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(self.adaLN_modulation[-1].bias, 0)
        else:
            for block in self.blocks:
                nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
                nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        if self.zero_in:
            nn.init.constant_(self.input_layer.weight, 0)
            nn.init.constant_(self.input_layer.bias, 0)

        # Zero-out output layers:
        if self.zero_out:
            nn.init.constant_(self.out_layer.weight, 0)
            nn.init.constant_(self.out_layer.bias, 0)

    def forward(
        self,
        x: sp.SparseTensor,
        t: torch.Tensor,
        embed: Optional[torch.Tensor] = None,
        context: Union[Dict[str, torch.Tensor], Dict[str, sp.SparseTensor]] = None,
        causal: bool = False,
        controls: Optional[Dict[str, sp.SparseTensor]] = None,
        return_intermediates: bool = False,
    ) -> Union[sp.SparseTensor, Dict[str, sp.SparseTensor]]:
        h = self.input_layer(x)

        t_emb = self.t_embedder(t)
        if self.share_mod:
            t_emb = self.adaLN_modulation(t_emb)

        if embed is not None:
            h = h.replace(h.feats + embed)

        if self.pe_mode == "ape":
            h = h + self.pos_embedder(h.coords[:, 1:])

        intermediates = {}
        for idx, block in enumerate(self.blocks):
            h = block(h, t_emb, context, causal=causal)
            intermediates[str(idx)] = h
            if controls is not None and str(idx) in controls:
                h = h.replace(h.feats + controls[str(idx)].feats)

        h = h.replace(F.layer_norm(h.feats, h.feats.shape[-1:]))
        h = self.out_layer(h.type(x.dtype))

        if return_intermediates:
            return intermediates
        else:
            return h
