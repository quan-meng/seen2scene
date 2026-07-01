import torch
from einops import rearrange
try:
    from flash_attn import flash_attn_varlen_func
except ImportError:
    flash_attn_varlen_func = None
from ..ops import (
    spatial_selection_attention,
    get_block_score,
    sparse_window_attention,
)
from .compression_block import SparseDownBlock3d_v1, SparseDownBlock3d_v2
from seen2scene.models import sparse as sp


class SpatialSparseAttention(torch.nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_q_heads: int,
        num_kv_heads: int,
        head_dim: int,
        compression_block_size: int,
        selection_block_size: int,
        topk: int,
        window_size: int,
        shift_window: int,
        resolution: int = 64,
        compression_version: str = "v2",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.compression_block_size = compression_block_size
        self.selection_block_size = selection_block_size
        self.topk = topk
        self.window_size = window_size
        self.shift_window = shift_window
        self.resolution = resolution

        # qkv proj and o proj
        self.proj_q = sp.SparseLinear(hidden_size, num_q_heads * head_dim, bias=False)
        self.proj_k = sp.SparseLinear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.proj_v = sp.SparseLinear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.proj_o = torch.nn.Linear(num_q_heads * head_dim, hidden_size, bias=False)

        # ssa parameteres
        if compression_version == "v1":
            compression_block = SparseDownBlock3d_v1
        elif compression_version == "v2":
            compression_block = SparseDownBlock3d_v2
        else:
            raise NotImplementedError("only support v1 or v2 compression block")
        self.compression_key = compression_block(
            num_kv_heads * head_dim,
            num_kv_heads * head_dim,
            factor=compression_block_size,
        )
        self.compression_value = compression_block(
            num_kv_heads * head_dim,
            num_kv_heads * head_dim,
            factor=compression_block_size,
        )
        self.intra_block_pe = torch.nn.Parameter(
            torch.zeros(
                compression_block_size,
                compression_block_size,
                compression_block_size,
                num_kv_heads * head_dim,
            )
        )

        # gate function
        self.gate = torch.nn.Sequential(
            sp.SparseLinear(hidden_size, 3, bias=False),
            sp.SparseSigmoid(),
        )

    def sparse3d_compression(self, x, key=True):
        _, num_heads, num_dim = x.feats.shape
        x = x.replace(x.feats.view(-1, num_heads * num_dim))
        if key:
            coords = x.coords
            intra_block_coords = coords[..., 1:] % self.compression_block_size
            intra_block_pos = self.intra_block_pe[
                intra_block_coords[:, 0],
                intra_block_coords[:, 1],
                intra_block_coords[:, 2],
            ].to(x.dtype)
            x = x.replace(x.feats + intra_block_pos)
            y = self.compression_key(x)
        else:
            y = self.compression_value(x)
        y = y.replace(y.feats.view(-1, num_heads, num_dim))
        return y

    def forward(self, x: sp.SparseTensor, **unused_kwargs):
        # dtype and shape check
        assert x.shape[-1] == self.hidden_size
        assert self.selection_block_size % self.compression_block_size == 0
        # qkv proj
        q = x.replace(self.proj_q(x).feats.view(-1, self.num_q_heads, self.head_dim))
        k = x.replace(self.proj_k(x).feats.view(-1, self.num_kv_heads, self.head_dim))
        v = x.replace(self.proj_v(x).feats.view(-1, self.num_kv_heads, self.head_dim))

        # compression attention
        compressed_k = self.sparse3d_compression(k, key=True)
        compressed_v = self.sparse3d_compression(v, key=False)

        compressed_cu_seqlens = (
            torch.tensor(
                [s.start for s in compressed_v.layout]
                + [
                    s.stop
                    for s in compressed_v.layout
                    if s.stop not in [s.start for s in compressed_v.layout]
                ]
            )
            .to(compressed_v.device)
            .to(torch.int32)
        )
        compressed_seqlens = compressed_cu_seqlens[1:] - compressed_cu_seqlens[:-1]

        cu_seqlens = (
            torch.tensor(
                [s.start for s in x.layout]
                + [
                    s.stop
                    for s in x.layout
                    if s.stop not in [s.start for s in x.layout]
                ]
            )
            .to(x.device)
            .to(torch.int32)
        )
        seqlens = cu_seqlens[1:] - cu_seqlens[:-1]

        compressed_attn_output, lse, _ = flash_attn_varlen_func(
            q.feats,
            compressed_k.feats,
            compressed_v.feats,
            cu_seqlens,
            compressed_cu_seqlens,
            seqlens.max().item(),
            compressed_seqlens.max().item(),
            causal=False,
            return_attn_probs=True,
        )

        with torch.no_grad():
            block_topk, cu_seqblocks, cu_block_include_tokens = get_block_score(
                q,
                compressed_k,
                lse,
                self.resolution,
                self.compression_block_size,
                self.selection_block_size,
                self.topk,
                cu_seqlens,
                compressed_cu_seqlens,
                seqlens,
                compressed_seqlens,
                None,
            )

        # spatial selection attention
        selection_attn_output = spatial_selection_attention(
            q.feats,
            k.feats,
            v.feats,
            block_topk,
            cu_seqblocks,
            cu_block_include_tokens,
            self.selection_block_size,
            cu_seqlens,
            None,
        )

        # window attention
        window_attn_output = sparse_window_attention(
            q,
            k,
            v,
            window_size=self.window_size,
            shift_window=self.shift_window,
        ).feats

        # gate average
        gate = self.gate(x).feats
        attn_output = (
            gate[:, 0:1, None] * compressed_attn_output
            + gate[:, 1:2, None] * selection_attn_output
            + gate[:, 2:3, None] * window_attn_output
        )

        # rearrange and output proj
        attn_output = rearrange(attn_output, "n h d -> n (h d)")
        attn_output = self.proj_o(attn_output)

        return x.replace(attn_output)
