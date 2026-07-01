import torch
from torch import nn
import numpy as np
import fvdb.nn as fvnn
from typing import Optional


class Embedding(nn.Module):
    def __init__(self, in_dim, N_freqs, logscale=True):
        """
        Defines a function that embeds x to (x, sin(2^k x), cos(2^k x), ...)
        in_dim: number of input channels (3 for both xyz and direction)
        """
        super(Embedding, self).__init__()
        self.N_freqs = N_freqs
        self.in_dim = in_dim
        self.funcs = [torch.sin, torch.cos]
        self.out_dim = in_dim * (len(self.funcs) * N_freqs + 1)

        if logscale:
            self.freq_bands = 2 ** torch.linspace(0, N_freqs - 1, N_freqs)
        else:
            self.freq_bands = torch.linspace(1, 2 ** (N_freqs - 1), N_freqs)

    def forward(self, x: torch.Tensor, dim: int = -1):
        """
        Embeds x to (x, sin(2^k x), cos(2^k x), ...)
        Different from the paper, "x" is also in the output
        See https://github.com/bmild/nerf/issues/12
        Inputs:
            x: (B, self.in_dim)
        Outputs:
            out: (B, self.out_dim)
        """
        out = [x]
        for freq in self.freq_bands:
            for func in self.funcs:
                out += [func(freq * x)]

        return torch.cat(out, dim=dim)


class Embedding_fVDB(nn.Module):
    def __init__(
        self,
        in_dim: int,
        data_n_freqs: int = -1,
        ijk_n_freqs: int = -1,
        resolution: Optional[int] = None,
        logscale=True,
    ):
        """
        Defines a function that embeds x to (x, sin(2^k x), cos(2^k x), ...)
        in_dim: number of input channels (3 for both xyz and direction)
        """
        super(Embedding_fVDB, self).__init__()
        self.ijk_n_freqs = ijk_n_freqs
        self.data_n_freqs = data_n_freqs
        self.in_dim = in_dim
        self.funcs = [torch.sin, torch.cos]
        self.out_dim = in_dim
        self.resolution = resolution
        if ijk_n_freqs > 0:
            self.out_dim += 3 * (len(self.funcs) * ijk_n_freqs + 1)
            if logscale:
                self.ijk_freq_bands = 2 ** np.linspace(0, ijk_n_freqs - 1, ijk_n_freqs)
            else:
                self.ijk_freq_bands = np.linspace(
                    1, 2 ** (ijk_n_freqs - 1), ijk_n_freqs
                )

        if data_n_freqs > 0:
            self.out_dim += in_dim * len(self.funcs) * data_n_freqs
            if logscale:
                self.data_freq_bands = 2 ** np.linspace(
                    0, data_n_freqs - 1, data_n_freqs
                )
            else:
                self.data_freq_bands = np.linspace(
                    1, 2 ** (data_n_freqs - 1), data_n_freqs
                )

    def forward(self, x: fvnn.VDBTensor) -> fvnn.VDBTensor:
        """
        Embeds x to (x, sin(2^k x), cos(2^k x), ...)
        Different from the paper, "x" is also in the output
        See https://github.com/bmild/nerf/issues/12
        Inputs:
            x: VDBTensor: (B, in_dim)
        Outputs:
            out: VDBTensor: (B, out_dim)
        """
        out = [x.data.jdata]
        if self.data_n_freqs > 0:
            for freq in self.data_freq_bands:
                for func in self.funcs:
                    out += [func(freq * x.data.jdata)]

        if self.ijk_n_freqs > 0:
            dtype = x.data.jdata.dtype
            jdata = x.grid.ijk.jdata.to(dtype).clone() / (self.resolution - 1)
            out += [jdata]
            for freq in self.ijk_freq_bands:
                for func in self.funcs:
                    out += [func(freq * jdata)]

        out = torch.cat(out, dim=-1)

        return fvnn.VDBTensor(x.grid, x.grid.jagged_like(out), x.kmap)
