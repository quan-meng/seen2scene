import torch.nn as nn
import fvdb.nn as fvnn
import torch
import fvdb
from fvdb.nn import VDBTensor
from typing import Optional, Literal
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint


class ConvBlock(nn.Sequential):
    def __init__(
        self, in_channels: int, out_channels: int, order: str, num_groups: int
    ):
        super().__init__()
        for i, char in enumerate(order):
            if char == "r":
                self.add_module("ReLU", fvnn.ReLU(inplace=True))
            elif char == "s":
                self.add_module("SiLU", fvnn.SiLU(inplace=True))
            elif char == "c":
                self.add_module(
                    "Conv",
                    fvnn.SparseConv3d(
                        in_channels, out_channels, 3, 1, bias="g" not in order
                    ),
                )
            elif char == "g":
                num_channels = in_channels if i < order.index("c") else out_channels
                if num_channels < num_groups:
                    num_groups = 1
                self.add_module(
                    "GroupNorm",
                    fvnn.GroupNorm(
                        num_groups=num_groups, num_channels=num_channels, affine=True
                    ),
                )
            else:
                raise NotImplementedError


class SparseHead(nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        order,
        num_groups,
        enhanced: Optional[Literal["tanh", "sigmoid"]] = None,
    ):
        super().__init__()
        self.add_module(
            "SingleConv", ConvBlock(in_channels, in_channels, order, num_groups)
        )
        mid_channels = in_channels
        if out_channels > mid_channels:
            mid_channels = out_channels

        if enhanced is None:
            self.add_module(
                "OutConv", fvnn.SparseConv3d(in_channels, out_channels, 1, bias=True)
            )
        elif enhanced == "tanh":
            self.add_module(
                "OutConv", fvnn.SparseConv3d(in_channels, out_channels, 1, bias=True)
            )
            self.add_module("Tanh", fvnn.Tanh())
        elif enhanced == "sigmoid":
            self.add_module(
                "OutConv", fvnn.SparseConv3d(in_channels, out_channels, 1, bias=True)
            )
            self.add_module("Sigmoid", fvnn.Sigmoid())
        else:
            raise NotImplementedError


class SparseResBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        order: str,
        num_groups: int,
        encoder: bool,
        pooling=None,
        use_checkpoint: bool = False,
        pooling_factor=[2, 2, 2],
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint

        self.use_pooling = pooling is not None and encoder

        if encoder:
            conv1_in_channels = in_channels
            conv1_out_channels = out_channels // 2
            if conv1_out_channels < in_channels:
                conv1_out_channels = in_channels
            conv2_in_channels, conv2_out_channels = conv1_out_channels, out_channels
            if pooling == "max":
                self.maxpooling = fvnn.MaxPool(pooling_factor)
        else:
            conv1_in_channels, conv1_out_channels = in_channels, out_channels
            conv2_in_channels, conv2_out_channels = out_channels, out_channels

        self.conv1 = ConvBlock(conv1_in_channels, conv1_out_channels, order, num_groups)
        self.conv2 = ConvBlock(conv2_in_channels, conv2_out_channels, order, num_groups)

        self.skip_connection = None
        if conv1_in_channels != conv2_out_channels:
            self.skip_connection = fvnn.SparseConv3d(
                conv1_in_channels, conv2_out_channels, 1, 1
            )

    def _forward(self, input, hash_tree=None, feat_depth: Optional[int] = None):
        if self.use_pooling:
            if hash_tree is not None:
                input = self.maxpooling(input, hash_tree[feat_depth])
            else:
                input = self.maxpooling(input)

        h = input
        h = self.conv1(h)
        h = self.conv2(h)
        if self.skip_connection is not None:
            h = self.skip_connection(input) + h
        else:
            h = fvnn.VDBTensor(
                h.grid, h.grid.jagged_like(h.data.jdata + input.data.jdata), h.kmap
            )

        return h

    def forward(self, input, hash_tree=None, feat_depth: Optional[int] = None):
        if self.use_checkpoint and self.training:
            input = checkpoint.checkpoint(
                self._forward, input, hash_tree, feat_depth, use_reentrant=False
            )
        else:
            input = self._forward(input, hash_tree, feat_depth)
        return input


class SparseDoubleConv(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        order: str,
        num_groups: int,
        encoder: bool,
        pooling=None,
        use_checkpoint: bool = False,
        pooling_factor=[2, 2, 2],
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        if encoder:
            conv1_in_channels = in_channels
            conv1_out_channels = out_channels // 2
            if conv1_out_channels < in_channels:
                conv1_out_channels = in_channels
            conv2_in_channels, conv2_out_channels = conv1_out_channels, out_channels
            if pooling == "max":
                self.add_module("MaxPool", fvnn.MaxPool(pooling_factor))
        else:
            conv1_in_channels, conv1_out_channels = in_channels, out_channels
            conv2_in_channels, conv2_out_channels = out_channels, out_channels

        self.add_module(
            "SingleConv1",
            ConvBlock(conv1_in_channels, conv1_out_channels, order, num_groups),
        )
        self.add_module(
            "SingleConv2",
            ConvBlock(conv2_in_channels, conv2_out_channels, order, num_groups),
        )

    def _forward(self, input, hash_tree=None, feat_depth: Optional[int] = None):
        for module in self:
            if module._get_name() == "MaxPool" and hash_tree is not None:
                input = module(input, hash_tree[feat_depth])
            else:
                input = module(input)
        return input

    def forward(self, input, hash_tree=None, feat_depth: Optional[int] = None):
        if self.use_checkpoint and self.training:
            input = checkpoint.checkpoint(  
                self._forward, input, hash_tree, feat_depth, use_reentrant=False
            )
        else:
            input = self._forward(input, hash_tree, feat_depth)
        return input


class AttentionBlock(nn.Module):
    """
    A for loop version with flash attention
    """

    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        use_checkpoint=False,
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.use_checkpoint = use_checkpoint
        self.norm = fvnn.GroupNorm(32, channels)
        self.qkv = fvnn.Linear(channels, channels * 3)
        self.proj_out = fvnn.Linear(channels, channels)

    def _attention(self, qkv: torch.Tensor):
        # conduct attention for each batch
        length, width = qkv.shape
        assert width % (3 * self.num_heads) == 0
        ch = width // (3 * self.num_heads)
        qkv = qkv.reshape(length, self.num_heads, 3 * ch).unsqueeze(0)
        qkv = qkv.permute(0, 2, 1, 3)  # (1, num_heads, length, 3 * ch)
        q, k, v = qkv.chunk(3, dim=-1)  # (1, num_heads, length, ch)
        with torch.backends.cuda.sdp_kernel(enable_math=False):
            values = F.scaled_dot_product_attention(q, k, v)[
                0
            ]  # (1, num_heads, length, ch)
        values = values.permute(1, 0, 2)  # (length, num_heads, ch)
        values = values.reshape(length, -1)
        return values

    def attention(self, qkv: VDBTensor):
        values = []
        for batch_idx in range(qkv.grid.grid_count):
            values.append(self._attention(qkv.data[batch_idx].jdata))
        return fvdb.JaggedTensor(values)

    def forward(self, x: VDBTensor):
        return self._forward(x), None  # !: return None for feat_depth

    def _forward(self, x: VDBTensor):
        qkv = self.qkv(self.norm(x))
        feature = self.attention(qkv)
        feature = VDBTensor(x.grid, feature, x.kmap)
        feature = self.proj_out(feature)
        return feature + x


def struct_to_decision(struct_pred: fvnn.VDBTensor) -> fvdb.JaggedTensor:
    logits = struct_pred.data.jdata[:, 0]  # [N]

    category = torch.sigmoid(logits) > 0.5
    return struct_pred.grid.jagged_like(category)
