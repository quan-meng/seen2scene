from typing import *
import torch
try:
    import flash_attn
except ImportError:
    flash_attn = None
from seen2scene.models.sparse import SparseTensor
from seen2scene.models.sparse.attention.windowed_attn import calc_window_partition


def sparse_window_attention(
    q: SparseTensor,
    k: SparseTensor,
    v: SparseTensor,
    window_size: int,
    shift_window: Tuple[int, int, int] = (0, 0, 0)
) -> SparseTensor:
    """
    Apply windowed scaled dot product self attention to a sparse tensor.

    Args:
        q (SparseTensor): [N, *, H_q, C] sparse tensor containing query.
        k (SparseTensor): [N, *, H_kv, C] sparse tensor containing key.
        v (SparseTensor): [N, *, H_kv, C] sparse tensor containing value.
        window_size (int): The window size to use.
        shift_window (Tuple[int, int, int]): The shift of serialized coordinates.
        shift (int): The shift to use.
    """

    serialization_spatial_cache_name = f'window_partition_{window_size}_{shift_window}'
    serialization_spatial_cache = q.get_spatial_cache(serialization_spatial_cache_name)
    if serialization_spatial_cache is None:
        fwd_indices, bwd_indices, seq_lens, seq_batch_indices = calc_window_partition(q, window_size, shift_window)
        q.register_spatial_cache(serialization_spatial_cache_name, (fwd_indices, bwd_indices, seq_lens, seq_batch_indices))
    else:
        fwd_indices, bwd_indices, seq_lens, seq_batch_indices = serialization_spatial_cache

    M = fwd_indices.shape[0]
    T = q.feats.shape[0]
    H = q.feats.shape[1]
    H_kv = k.feats.shape[1]
    C = q.feats.shape[2]
    q_feats = q.feats[fwd_indices]      # [M, H, C]
    k_feats = k.feats[fwd_indices]
    v_feats = v.feats[fwd_indices]

    if all([seq_len == window_size for seq_len in seq_lens]):
        B = len(seq_lens)
        N = window_size
        q_feats = q_feats.reshape(B, N, H, C)
        k_feats = k_feats.reshape(B, N, H_kv, C)
        v_feats = v_feats.reshape(B, N, H_kv, C)
        out = flash_attn.flash_attn_func(q_feats, k_feats, v_feats)
        out = out.reshape(B * N, H, C)                              # [M, H, C]
    else:
        cu_seqlens = torch.cat([torch.tensor([0]), torch.cumsum(torch.tensor(seq_lens), dim=0)], dim=0) \
                    .to(q.device).int()
        out = flash_attn.flash_attn_varlen_func(q_feats, k_feats, v_feats, cu_seqlens, cu_seqlens, max(seq_lens), max(seq_lens))

    out = out[bwd_indices]      # [T, H, C]

    return q.replace(out)