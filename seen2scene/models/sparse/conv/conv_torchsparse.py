import torch
import torch.nn as nn
from .. import SparseTensor
from torchsparse.utils import make_ntuple


def sparseconv3d_func(
    input: SparseTensor,
    weight: torch.Tensor,
    kernel_size: int,
    stride: int = 1,
    dilation: int = 1,
    padding: int = 0,
    bias: torch.Tensor = None,
    training: bool = True,
):
    if "torchsparse" not in globals():
        import torchsparse
    stride = make_ntuple(stride, ndim=3)
    kernel_size = make_ntuple(kernel_size, ndim=3)
    _padding = make_ntuple(padding, 3)
    padding = ()
    for i in range(3):
        if kernel_size[i] % 2 == 1 and stride[i] == 1:
            padding += ((kernel_size[i] - 1) // 2,)
        else:
            padding += (_padding[i],)
    out = torchsparse.nn.functional.conv3d(
        input.data,
        weight,
        kernel_size=kernel_size,
        stride=stride,
        dilation=dilation,
        padding=padding,
        bias=bias,
        training=training,
    )
    spatial_range = out.spatial_range
    new_shape = [input.shape[0], weight.shape[1]]
    out = SparseTensor(
        out,
        shape=torch.Size(new_shape),
        layout=input.layout if all(s == 1 for s in stride) else None,
    )
    out._spatial_cache = input._spatial_cache
    out._scale = tuple([s * stride for s, stride in zip(input._scale, stride)])
    out.data.spatial_range = spatial_range
    return out


class SparseConv3d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        dilation=1,
        padding=0,
        bias=True,
        indice_key=None,
    ):
        super(SparseConv3d, self).__init__()
        if "torchsparse" not in globals():
            import torchsparse
        self.conv = torchsparse.nn.Conv3d(
            in_channels, out_channels, kernel_size, stride, padding, dilation, bias
        )

    def forward(self, x: SparseTensor) -> SparseTensor:
        out = self.conv(x.data)

        spatial_range = out.spatial_range

        new_shape = [x.shape[0], self.conv.out_channels]
        out = SparseTensor(
            out,
            shape=torch.Size(new_shape),
            layout=x.layout if all(s == 1 for s in self.conv.stride) else None,
        )
        out._spatial_cache = x._spatial_cache
        out._scale = tuple(
            [s * stride for s, stride in zip(x._scale, self.conv.stride)]
        )

        out.data.spatial_range = spatial_range

        return out


class SparseInverseConv3d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        dilation=1,
        bias=True,
        indice_key=None,
    ):
        super(SparseInverseConv3d, self).__init__()
        if "torchsparse" not in globals():
            import torchsparse
        self.conv = torchsparse.nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            0,
            dilation,
            bias,
            transposed=True,
        )

    def forward(self, x: SparseTensor) -> SparseTensor:
        out = self.conv(x.data)

        new_shape = [x.shape[0], self.conv.out_channels]
        out = SparseTensor(
            out,
            shape=torch.Size(new_shape),
            layout=x.layout if all(s == 1 for s in self.conv.stride) else None,
        )
        out._spatial_cache = x._spatial_cache
        out._scale = tuple(
            [s // stride for s, stride in zip(x._scale, self.conv.stride)]
        )

        return out
