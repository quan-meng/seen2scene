import torch.nn as nn
import math
import torch
import fvdb
from typing import *
import fvdb.nn as fvnn
from einops import rearrange
import torch.nn.functional as F

from .common import SparseDoubleConv, SparseResBlock, AttentionBlock
from seen2scene.tools.embedder_utils import Embedding_fVDB
from seen2scene.tools.log_utils import get_logger
from seen2scene.configs.dataset import Voxel
from seen2scene.models.trainer.common import get_instance_mask

logger = get_logger(__name__, debug="fvdb_encoder")


class DownLayers(nn.Module):
    def __init__(
        self,
        in_channels: int,
        n_features: List[int],
        is_add_dec: bool = True,
        num_res_blocks: int = 1,
        pooling: str = "max",
        use_checkpoint: bool = False,
        pooling_factor: List[int] = [2, 2, 2],
        order: str = "gcr",
        num_groups: int = 8,
        basic_block: Union[SparseDoubleConv, SparseResBlock] = SparseDoubleConv,
    ):
        super().__init__()
        self.in_channels = in_channels

        self.pos_enc = Embedding_fVDB(in_channels, data_n_freqs=10)
        self.pre_conv = fvnn.SparseConv3d(self.pos_enc.out_dim, n_features[0], 1, 1)

        num_blocks = len(n_features) - 1
        self.layers = nn.ModuleList()
        for layer_idx in range(num_blocks):
            block_in = n_features[layer_idx]
            block_out = n_features[layer_idx + 1]

            if is_add_dec:
                for i_block in range(num_res_blocks):
                    block = basic_block(
                        block_in,
                        block_out,
                        order,
                        num_groups,
                        True,  # if encoder branch
                        pooling if i_block == 0 else None,
                        use_checkpoint,
                        pooling_factor=pooling_factor,
                    )
                    block_in = block_out
                    self.layers.add_module(f"Enc{layer_idx}-Block{i_block}", block)
            else:
                block = basic_block(
                    block_in,
                    block_out,
                    order,
                    num_groups,
                    True,  # if encoder branch
                    pooling,
                    use_checkpoint,
                    pooling_factor=pooling_factor,
                )
                self.layers.add_module(f"Enc{layer_idx}", block)

        self.out_channels = block_out

    def forward(
        self, x: fvnn.VDBTensor, hash_tree: Optional[list] = None
    ) -> Tuple[fvnn.VDBTensor, dict]:
        x = self.pos_enc(x)
        x = self.pre_conv(x)

        feat_depth = 0
        for module in self.layers:
            x = module(x, hash_tree)
            feat_depth += 1

        return x


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        factor: int,
        num_res_blocks: int,
        out_channels: Optional[int] = None,
        order: str = "gcr",
        use_residual: bool = True,
        f_maps: int = 32,
        is_add_dec: bool = True,
        num_groups: int = 8,
        pooling: str = "max",
        use_checkpoint: bool = False,
        use_attention: bool = False,
        double_z: bool = False,
        **unused_kwargs,
    ):
        super().__init__()
        self.factor = factor

        if not use_residual:
            basic_block = SparseDoubleConv
        else:
            basic_block = SparseResBlock
        num_blocks = int(math.log2(factor))
        self.band_enc = DownLayers(
            in_channels=in_channels,
            n_features=[f_maps * (2**k) for k in range(num_blocks + 1)],
            is_add_dec=is_add_dec,
            num_res_blocks=num_res_blocks,
            pooling=pooling,
            use_checkpoint=use_checkpoint,
            order=order,
            num_groups=num_groups,
            basic_block=basic_block,
        )

        middle_channels = self.band_enc.out_channels

        self.empty_token = nn.Parameter(torch.randn(1, middle_channels) * 0.01)

        # Bottleneck
        self.pre_kl_bottleneck = nn.ModuleList()
        self.pre_kl_bottleneck.add_module(
            f"pre_kl_bottleneck_0",
            basic_block(
                middle_channels,
                middle_channels,
                order,
                num_groups,
                False,
                use_checkpoint=use_checkpoint,
            ),
        )
        if use_attention:
            self.pre_kl_bottleneck.add_module(
                f"pre_kl_attention",
                AttentionBlock(middle_channels, use_checkpoint=use_checkpoint),
            )
        if not use_residual:
            self.pre_kl_bottleneck.add_module(
                f"pre_kl_bottleneck_1",
                basic_block(
                    middle_channels,
                    out_channels * (2 if double_z else 1),
                    order,
                    num_groups,
                    False,
                    use_checkpoint=use_checkpoint,
                ),
            )
        else:
            self.pre_kl_bottleneck.add_module(
                f"pre_kl_bottleneck_1",
                basic_block(
                    middle_channels,
                    middle_channels,
                    order,
                    num_groups,
                    False,
                    use_checkpoint=use_checkpoint,
                ),
            )
            self.pre_kl_bottleneck.add_module(
                f"pre_kl_bottleneck_gn",
                fvnn.GroupNorm(
                    num_groups=num_groups, num_channels=middle_channels, affine=True
                ),
            )
            self.pre_kl_bottleneck.add_module(
                f"pre_kl_bottleneck_2",
                fvnn.SparseConv3d(
                    middle_channels,
                    out_channels * (2 if double_z else 1),
                    3,
                    1,
                    bias=True,
                ),
            )

        self.initialize_weights()

    def initialize_weights(self) -> None:
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, fvnn.SparseConv3d):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

    @torch.no_grad()
    def structure_encoding(
        self,
        struct: torch.Tensor,
        voxel_sizes: torch.Tensor,
        origins: torch.Tensor,
        mask_unknown: bool = True,
        keep_region: Optional[List[str]] = None,
        object_bboxes: Optional[List[torch.Tensor]] = None,
    ) -> fvdb.GridBatch:
        struct_down = F.max_pool3d(
            struct.to(torch.float16), kernel_size=self.factor, stride=self.factor
        )
        struct_down = rearrange(struct_down, "b c h w d -> b h w d c").contiguous()
        struct_down = fvnn.vdbtensor_from_dense(
            struct_down, [0, 0, 0], voxel_sizes=voxel_sizes, origins=origins
        )

        surf_ratio = (struct_down.data.jdata == Voxel.BAND).float().mean()
        empty_ratio = (struct_down.data.jdata == Voxel.EMPTY).float().mean()
        stat_dict = {
            "surf_token_ratio": surf_ratio,
            "empty_token_ratio": empty_ratio,
            "known_token_ratio": surf_ratio + empty_ratio,
        }
        if mask_unknown:
            mask = struct_down.data.jdata[:, 0] != Voxel.UNKNOWN  # [N]

            xyzs = struct_down.grid.grid_to_world(
                struct_down.grid.ijk.float()
            )  # [N, 3]
            xyzs_list = [xyzs_i.jdata for xyzs_i in xyzs]
            ins_mask = get_instance_mask(xyzs_list, object_bboxes)

            if keep_region is not None:
                neighbor_idxs = struct_down.grid.neighbor_indexes(
                    struct_down.grid.ijk, 2
                )  # [N, g, g, g]
                neighbor_values = struct_down.data.jdata[neighbor_idxs.jdata]  # [N, g, g, g]
                neighbor_values = torch.where(
                    neighbor_values == Voxel.EMPTY,
                    1.0,
                    torch.where(neighbor_values == Voxel.BAND, 0.5, 0),
                )
                never_scan_mask = neighbor_values.mean(dim=[-1, -2, -3, -4]) < 1e-2  # [N]
                ins_mask = ins_mask.any(dim=-1)  # [N_total]
                mask = mask | ins_mask

            ijks = struct_down.grid.ijk.rmask(mask)
            grid = fvdb.gridbatch_from_ijk(
                ijks, voxel_sizes=voxel_sizes, origins=origins
            )
        else:
            grid = struct_down.grid

        return grid, stat_dict

    def forward(
        self,
        x: fvnn.VDBTensor,
        structure: torch.Tensor,
        hash_tree: Optional[list] = None,
        mask_unknown: bool = True,
        keep_region: Optional[List[str]] = None,
        object_bboxes: Optional[List[torch.Tensor]] = None,
    ) -> Union[fvnn.VDBTensor, dict]:
        x = self.band_enc(x, hash_tree)

        voxel_sizes = x.grid.voxel_sizes
        origins = x.grid.origins

        grid, stat_dict = self.structure_encoding(
            structure,
            voxel_sizes=voxel_sizes,
            origins=origins,
            mask_unknown=mask_unknown,
            keep_region=keep_region,
            object_bboxes=object_bboxes,
        )

        # fVDB CUDA kernels crash on 0-voxel batch elements.
        # structure_encoding can produce these when a patch is entirely UNKNOWN.
        # Inject a dummy voxel at an out-of-range coordinate so that band_mask=0
        # and it receives empty_token features → decoder produces no geometry.
        has_empty = any(
            grid.num_voxels_at(i) == 0 for i in range(grid.grid_count)
        )
        if has_empty:
            all_ijks = []
            for i in range(grid.grid_count):
                if grid.num_voxels_at(i) == 0:
                    # Use out-of-range coord so ijk_to_index returns -1 → band_mask=0
                    all_ijks.append(torch.full(
                        (1, 3), -1, device=grid.device, dtype=torch.int32
                    ))
                else:
                    all_ijks.append(grid.ijk[i].jdata)
            grid = fvdb.gridbatch_from_ijk(
                fvdb.JaggedTensor(all_ijks),
                voxel_sizes=voxel_sizes,
                origins=origins,
            )

        jagged_band = grid.fill_from_grid(x.data, x.grid, 0.0)

        band_mask = (x.grid.ijk_to_index(grid.ijk).jdata != -1)[:, None].float()
        jagged_band = grid.jagged_like(
            band_mask * jagged_band.jdata + (1 - band_mask) * self.empty_token
        )
        x = fvnn.VDBTensor(grid, jagged_band)

        for module in self.pre_kl_bottleneck:
            x = module(x)

        return x, band_mask[:, 0], stat_dict
