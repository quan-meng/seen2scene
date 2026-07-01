import torch
import math
import torch.nn as nn
import fvdb.nn as fvnn
import numpy as np
from collections import OrderedDict
from typing import Optional, Tuple, Dict, Any

from seen2scene.models.fvdb.common import (
    SparseHead,
    SparseResBlock,
    SparseDoubleConv,
    AttentionBlock,
    struct_to_decision,
)
from seen2scene.tools.log_utils import get_logger

logger = get_logger(file_name=__file__, debug="fvdb_decoder")


# f(t) = p + (1 - p) \cdot \left(\frac{t}{6}\right)^2
def compute_cutoff_rate(t: int, max_down_time: int = 6, percentage: float = 0.15):
    return percentage + (1 - percentage) * ((t / max_down_time) ** 2)


class Decoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        factor: int = 4,
        f_maps: int = 32,
        order: str = "gcr",
        num_groups: int = 32,
        num_res_blocks: int = 1,
        use_residual: bool = True,
        use_checkpoint: bool = False,
        num_semantic_classes: int = 23,
        with_color_branch: bool = False,
        with_semantic_branch: bool = False,
        use_attention: bool = False,
        is_add_dec: bool = True,
        unstable_cutoff_threshold: float = 0.5,
        max_down_time: int = 6,
        **unused_kwargs,
    ):
        super().__init__()
        self.unstable_cutoff_threshold = unstable_cutoff_threshold
        self.is_add_dec = is_add_dec
        self.num_blocks = int(math.log2(factor)) + 1
        n_features = [f_maps * (2**k) for k in range(self.num_blocks)]
        self.n_features = n_features
        self.max_down_time = max_down_time

        if not use_residual:
            basic_block = SparseDoubleConv
        else:
            basic_block = SparseResBlock

        self.pre_conv = fvnn.SparseConv3d(in_channels, n_features[-1], 3, 1)

        self.post_kl_bottleneck = nn.ModuleList()
        self.post_kl_bottleneck.add_module(
            f"post_kl_bottleneck_0",
            basic_block(
                n_features[-1],
                n_features[-1],
                order,
                num_groups,
                False,
                use_checkpoint=use_checkpoint,
            ),
        )
        if use_attention:
            self.post_kl_bottleneck.add_module(
                f"post_kl_attention",
                AttentionBlock(n_features[-1], use_checkpoint=use_checkpoint),
            )
        self.post_kl_bottleneck.add_module(
            f"post_kl_bottleneck_1",
            basic_block(
                n_features[-1],
                n_features[-1],
                order,
                num_groups,
                False,
                use_checkpoint=use_checkpoint,
            ),
        )

        # Decoder
        self.upsamplers = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.struct_convs = nn.ModuleList()
        self.cutoff_rates = []

        for layer_idx in range(-1, -self.num_blocks - 1, -1):
            self.struct_convs.add_module(
                f"Struct{layer_idx}",
                SparseHead(n_features[layer_idx], 1, order, num_groups),
            )
            self.cutoff_rates.append(
                compute_cutoff_rate(
                    self.num_blocks + layer_idx,
                    self.max_down_time,
                    self.unstable_cutoff_threshold,
                )
            )
            if layer_idx < -1:
                block_in = n_features[layer_idx + 1]
                block_out = n_features[layer_idx]

                if is_add_dec:
                    block = nn.ModuleList()
                    for i_block in range(num_res_blocks):
                        block.append(
                            basic_block(
                                block_in,
                                block_out,
                                order,
                                num_groups,
                                False,  # if decoder branch
                                None,
                                use_checkpoint,
                            )
                        )
                        block_in = block_out
                    self.decoders.add_module(f"Dec{layer_idx}", block)
                else:
                    self.decoders.add_module(
                        f"Dec{layer_idx}",
                        basic_block(
                            block_in,
                            block_out,
                            order,
                            num_groups,
                            False,
                            None,
                            use_checkpoint,
                        ),
                    )

                self.upsamplers.add_module(
                    f"Up{layer_idx}", fvnn.UpsamplingNearest((2, 2, 2))
                )

        self.band_filter = fvnn.UpsamplingNearest(1)

        self.geometry_head = SparseHead(
            n_features[0], 1, order, num_groups, enhanced="tanh"
        )

        self.with_semantic_branch = with_semantic_branch
        if with_semantic_branch:
            self.semantic_head = SparseHead(
                n_features[0], num_semantic_classes, order, num_groups
            )
        self.with_color_branch = with_color_branch
        if with_color_branch:
            self.color_head = SparseHead(n_features[0], 3, order, num_groups)

        # self.initialize_weights()

    @torch.no_grad()
    def initialize_weights(self) -> None:
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, fvnn.SparseConv3d):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

    def forward(
        self,
        x: fvnn.VDBTensor,
        unstable_cutoff: bool = False,
        truncation: Optional[float] = None,
        voxel_bound: Optional[Tuple[int, int, int]] = None,
        decision_tree: Optional[Dict[int, fvnn.VDBTensor]] = None,
    ) -> Dict[str, Any]:
        out = {"struct_feats": OrderedDict()}
        state_dict = OrderedDict()

        x = self.pre_conv(x)
        for module in self.post_kl_bottleneck:
            x = module(x)

        struct_decision = None
        feat_depth = self.num_blocks - 1
        for block, upsampler, struct_conv, cutoff_rate_threshold in zip(
            [None] + list(self.decoders),
            [None] + list(self.upsamplers),
            self.struct_convs,
            self.cutoff_rates,
        ):
            if block is not None:
                x = upsampler(x, struct_decision)
                x = block(x)

            out["struct_feats"][feat_depth] = struct_conv(x)

            if decision_tree is None:
                # prune empty voxels
                struct_decision = struct_to_decision(out["struct_feats"][feat_depth])
            else:
                struct_decision = decision_tree[feat_depth]

                assert torch.allclose(
                    out["struct_feats"][feat_depth].grid.origins,
                    struct_decision.grid.origins,
                )
                assert torch.allclose(
                    out["struct_feats"][feat_depth].grid.voxel_sizes,
                    struct_decision.grid.voxel_sizes,
                )

                struct_decision = struct_decision.grid.ijk_to_index(
                    out["struct_feats"][feat_depth].grid.ijk
                )
                struct_decision = struct_decision.jagged_like(
                    struct_decision.jdata != -1
                )

            if feat_depth != self.num_blocks - 1 and unstable_cutoff:
                current_voxel_bound = [
                    res * 2 ** (self.num_blocks - 1 - feat_depth) for res in voxel_bound
                ]
                max_voxel_count = np.prod(current_voxel_bound)
                current_ratio = struct_decision.jdata.sum() / (
                    max_voxel_count * x.grid.grid_count
                )
                state_dict[f"voxel_ratio_{feat_depth}"] = current_ratio
                if current_ratio > cutoff_rate_threshold and self.training:
                    logger.warning(
                        f"cut off at depth {feat_depth} with ratio {current_ratio}"
                    )
                    struct_decision = struct_decision.jagged_like(
                        torch.zeros_like(struct_decision.jdata)
                    )

            feat_depth -= 1

        x = self.band_filter(x, struct_decision)
        out["geometry"] = self.geometry_head(x) * truncation

        # Fix: Check for empty grid before any head operations
        if x.grid.total_voxels > 0:
            if self.with_semantic_branch:
                out["semantic"] = self.semantic_head(x)
            if self.with_color_branch:
                out["color"] = self.color_head(x)

        return out, state_dict
