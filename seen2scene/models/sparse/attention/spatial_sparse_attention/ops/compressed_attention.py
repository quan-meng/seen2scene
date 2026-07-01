# Copyright 2025 Xunhao Lai.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ---------------------------------------------------------------------
# Copyright 2025 Shuang Wu
# adapted from https://github.com/XunhaoLai/native-sparse-attention-triton/blob/main/native_sparse_attention/ops/triton/compressed_attention.py

import math
import torch
from copy import deepcopy
import triton
import triton.language as tl
from seen2scene.models import sparse as sp


@triton.jit
def score_kernel(
    q_ptr,
    k_ptr,
    lse_ptr,
    s_ptr,
    # seqlens
    cu_seqlens_q,
    cu_seqlens_k,
    # shape
    NUM_KV_HEADS,
    NUM_SHARE_Q_HEADS,
    HEAD_DIM,
    # sm_scale
    sm_scale,
    # stride
    stride_qn,
    stride_qh,
    stride_qd,
    stride_kn,
    stride_kh,
    stride_kd,
    stride_lh,
    stride_ln,
    stride_sh,
    stride_sq,
    stride_sk,
    # META parameters
    BLOCK_SIZE_Q: tl.constexpr,  # q block size
    BLOCK_SIZE_K: tl.constexpr,  # k block size
    BLOCK_SIZE_D: tl.constexpr,
):
    qk_scale = sm_scale * 1.44269504
    # get batch id and head id
    pid_bkh = tl.program_id(0)
    pid_b = pid_bkh // NUM_KV_HEADS
    pid_kh = pid_bkh % NUM_KV_HEADS
    pid_q = tl.program_id(1)
    pid_k = tl.program_id(2)
    # get q k start and len after rmpad
    q_start = tl.load(cu_seqlens_q + pid_b)
    q_len = tl.load(cu_seqlens_q + pid_b + 1) - q_start
    k_start = tl.load(cu_seqlens_k + pid_b)
    k_len = tl.load(cu_seqlens_k + pid_b + 1) - k_start
    if pid_q * BLOCK_SIZE_Q >= q_len or pid_k * BLOCK_SIZE_K >= k_len:
        return
    # init k pointer and load k
    k_ptrs = tl.make_block_ptr(
        base=k_ptr + k_start * stride_kn + pid_kh * stride_kh,
        shape=(HEAD_DIM, k_len),
        strides=(stride_kd, stride_kn),
        offsets=(0, pid_k * BLOCK_SIZE_K),
        block_shape=(BLOCK_SIZE_D, BLOCK_SIZE_K),
        order=(0, 1),
    )
    k = tl.load(k_ptrs, boundary_check=(0, 1), padding_option="zero")
    # init score
    s = tl.zeros((BLOCK_SIZE_Q, BLOCK_SIZE_K), dtype=tl.float32)
    # loop over gqa heads
    for h in range(NUM_SHARE_Q_HEADS):
        pid_h = pid_kh * NUM_SHARE_Q_HEADS + h
        q_ptrs = tl.make_block_ptr(
            base=q_ptr + q_start * stride_qn + pid_h * stride_qh,
            shape=(q_len, HEAD_DIM),
            strides=(stride_qn, stride_qd),
            offsets=(pid_q * BLOCK_SIZE_Q, 0),
            block_shape=(BLOCK_SIZE_Q, BLOCK_SIZE_D),
            order=(1, 0),
        )
        lse_ptrs = tl.make_block_ptr(
            base=lse_ptr + q_start * stride_ln + pid_h * stride_lh,
            shape=(q_len, 1),
            strides=(stride_ln, stride_lh),
            offsets=(pid_q * BLOCK_SIZE_Q, 0),
            block_shape=(BLOCK_SIZE_Q, 1),
            order=(0, 1),
        )
        # load q and lse
        q = tl.load(q_ptrs, boundary_check=(0, 1), padding_option="zero")
        lse = tl.load(lse_ptrs, boundary_check=(0, 1), padding_option="zero")
        # compute qk
        qk = tl.zeros((BLOCK_SIZE_Q, BLOCK_SIZE_K), dtype=tl.float32)
        qk += tl.dot(q, k) * qk_scale
        # compute score
        s += tl.exp2(qk - lse)
    # save output
    s_ptrs = tl.make_block_ptr(
        base=s_ptr + pid_kh * stride_sh + q_start * stride_sq,
        shape=(q_len, k_len),
        strides=(stride_sq, stride_sk),
        offsets=(pid_q * BLOCK_SIZE_Q, pid_k * BLOCK_SIZE_K),
        block_shape=(BLOCK_SIZE_Q, BLOCK_SIZE_K),
        order=(1, 0),
    )
    tl.store(s_ptrs, s.to(s_ptr.dtype.element_ty), boundary_check=(0, 1))


def _get_attention_score(
    q: torch.Tensor,  # [total_query_len, num_q_heads, head_dim]
    k: torch.Tensor,  # [total_key_len, num_k_heads, head_dim]
    lse: torch.Tensor,  # [num_q_heads, total_query_len]
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    sm_scale: float,
) -> torch.Tensor:
    # dtype check
    assert q.dtype == torch.bfloat16 or q.dtype == torch.float16
    assert q.dtype == k.dtype
    assert cu_seqlens_q.dtype == torch.int32 and cu_seqlens_k.dtype == torch.int32
    assert lse.dtype == torch.float32
    # shape
    q_len, num_q_heads, head_dim = q.shape
    k_len, num_k_heads, head_dim = k.shape
    batch_size = cu_seqlens_q.shape[0] - 1
    assert q_len > k_len
    if sm_scale is None:
        sm_scale = 1 / math.sqrt(head_dim)
    # gqa
    assert num_q_heads % num_k_heads == 0
    num_share_q_heads = num_q_heads // num_k_heads
    # init score
    score = torch.zeros(
        num_k_heads, q_len, max_seqlen_k, dtype=torch.float32, device=q.device
    )
    # launch kernel
    grid = lambda META: (
        batch_size * num_k_heads,
        triton.cdiv(max_seqlen_q, META["BLOCK_SIZE_Q"]),
        triton.cdiv(max_seqlen_k, META["BLOCK_SIZE_K"]),
    )
    num_warps = 4 if head_dim <= 64 else 8
    num_stages = 3
    BLOCK_SIZE_Q = 128
    BLOCK_SIZE_K = 128
    BLOCK_SIZE_D = triton.next_power_of_2(head_dim)
    score_kernel[grid](
        q,
        k,
        lse,
        score,
        cu_seqlens_q,
        cu_seqlens_k,
        num_k_heads,
        num_share_q_heads,
        head_dim,
        sm_scale,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        lse.stride(0),
        lse.stride(1),
        score.stride(0),
        score.stride(1),
        score.stride(2),
        BLOCK_SIZE_Q=BLOCK_SIZE_Q,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        BLOCK_SIZE_D=BLOCK_SIZE_D,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return score


def get_block_score(
    q: sp.SparseTensor,
    compressed_k: sp.SparseTensor,
    lse: sp.SparseTensor,
    resolution: int,
    kernel_stride: int,
    block_size: int,
    topk: int,
    cu_seqlens: torch.Tensor,
    compressed_cu_seqlens: torch.Tensor,
    seqlens: torch.Tensor,
    compressed_seqlens: torch.Tensor,
    sm_scale: float = None,
) -> torch.Tensor:
    attn_score = _get_attention_score(
        q.feats,
        compressed_k.feats,
        lse.exp().log2(),
        cu_seqlens,
        compressed_cu_seqlens,
        seqlens.max().item(),
        compressed_seqlens.max().item(),
        sm_scale,
    )

    batch_size = len(cu_seqlens) - 1
    num_kv_head = attn_score.shape[0]
    block_res = resolution // block_size

    seqblocks, block_include_tokens = [], []
    block_topk = (
        torch.ones(
            (num_kv_head, cu_seqlens[-1], topk), device=q.device, dtype=torch.int32
        )
        * -1
    )

    q_coords = deepcopy(q.coords)
    for b in range(batch_size):
        q_start, q_end, q_len = cu_seqlens[b], cu_seqlens[b + 1], seqlens[b]

        compressed_k_start, compressed_k_end = (
            compressed_cu_seqlens[b],
            compressed_cu_seqlens[b + 1],
        )
        attn_score_b = attn_score[
            :, q_start:q_end, : (compressed_k_end - compressed_k_start)
        ]
        compressed_block_coords_b = deepcopy(
            compressed_k.coords[compressed_k_start:compressed_k_end]
        )
        if block_size == kernel_stride:
            score_block_b = attn_score_b
            real_topk = min(topk, compressed_k_end - compressed_k_start)
            block_topk_b = score_block_b.topk(real_topk, dim=-1).indices.sort(-1).values
            block_topk[:, q_start:q_end, :real_topk] = block_topk_b
        else:
            compressed_block_coords_b[:, 1:] = compressed_block_coords_b[:, 1:] // (
                block_size // kernel_stride
            )
            compressed_block_coords_flatten_b = (
                compressed_block_coords_b[:, 1] * block_res**2
                + compressed_block_coords_b[:, 2] * block_res
                + compressed_block_coords_b[:, 3]
            )
            score_block_b = torch.scatter_reduce(
                torch.zeros(
                    (num_kv_head, q_len, block_res**3),
                    device=attn_score_b.device,
                    dtype=attn_score_b.dtype,
                ),
                index=compressed_block_coords_flatten_b.long()
                .unsqueeze(0)
                .unsqueeze(0)
                .expand_as(attn_score_b),
                src=attn_score_b,
                reduce="sum",
                dim=2,
            )
            compressed_block_coords_flatten_unique_b = (
                compressed_block_coords_flatten_b.unique()
            )
            score_block_b = score_block_b[..., compressed_block_coords_flatten_unique_b]
            real_topk = min(topk, len(compressed_block_coords_flatten_unique_b))
            block_topk_b = score_block_b.topk(real_topk, dim=-1).indices.sort(-1).values
            block_topk[:, q_start:q_end, :real_topk] = block_topk_b

        block_coords_b = q_coords[q_start:q_end]
        block_coords_b[:, 1:] = block_coords_b[:, 1:] // block_size
        block_coords_flatten_b = (
            block_coords_b[:, 1] * block_res**2
            + block_coords_b[:, 2] * block_res
            + block_coords_b[:, 3]
        )
        block_bins_b = torch.histc(
            block_coords_flatten_b, bins=block_res**3, min=0, max=block_res**3 - 1
        )
        block_include_tokens.append(block_bins_b[block_bins_b > 0])
        seqblocks.append(len(block_include_tokens[-1]))
    seqblocks = torch.Tensor(seqblocks).to(attn_score.device)
    cu_seqblocks = torch.cat(
        [
            torch.zeros(1, dtype=torch.int32, device=attn_score.device),
            torch.cumsum(seqblocks, dim=0),
        ],
        dim=0,
    ).to(torch.int32)
    block_include_tokens = torch.cat(block_include_tokens)
    cu_block_include_tokens = torch.cat(
        [
            torch.zeros(1, dtype=torch.int32, device=attn_score.device),
            torch.cumsum(block_include_tokens, dim=0),
        ],
        dim=0,
    ).to(torch.int32)
    return block_topk.to(torch.int32), cu_seqblocks, cu_block_include_tokens
