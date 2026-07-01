import torch
import torch.nn as nn
from seen2scene.models import sparse as sp


class SparseDownBlock3d_v1(nn.Module):

    def __init__(
        self,
        channels: int,
        out_channels: int = None,
        factor: int = 2,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels

        self.act_layers = nn.Sequential(
            sp.SparseConv3d(self.out_channels, self.out_channels, 1, padding=0),
            sp.SparseSiLU(),
        )
        self.down = sp.SparseDownsample(factor)

    def forward(self, x: sp.SparseTensor) -> sp.SparseTensor:
        h = self.act_layers(x)
        h = self.down(h)
        return h


class SparseDownBlock3d_v2(nn.Module):

    def __init__(
        self,
        channels: int,
        out_channels: int = None,
        num_groups: int = 32,
        factor: int = 2,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels

        self.act_layers = nn.Sequential(
            sp.SparseGroupNorm32(num_groups, channels), sp.SparseSiLU()
        )

        self.down = sp.SparseDownsample(factor)
        self.out_layers = nn.Sequential(
            sp.SparseConv3d(channels, self.out_channels, 3, padding=1),
            sp.SparseGroupNorm32(num_groups, self.out_channels),
            sp.SparseSiLU(),
            sp.SparseConv3d(self.out_channels, self.out_channels, 3, padding=1),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = sp.SparseConv3d(channels, self.out_channels, 1)

    def forward(self, x: sp.SparseTensor) -> sp.SparseTensor:
        h = self.act_layers(x)
        h = self.down(h)
        x = self.down(x)

        # Store original dtype
        original_dtype = h.feats.dtype

        # Using pytorch lightning's autocast
        with torch.autocast("cuda", enabled=False):
            # Convert to float32 for sparse convolution operation
            h_float32 = h.replace(h.feats.float())
            x_float32 = x.replace(x.feats.float())

            h_float32 = self.out_layers(h_float32)
            skip_float32 = self.skip_connection(x_float32)

        # Convert back to original dtype
        h = h_float32.replace(h_float32.feats.to(original_dtype))
        skip = skip_float32.replace(skip_float32.feats.to(original_dtype))

        h = h + skip
        return h
