import torch
import fvdb
import torch.nn as nn
import fvdb.nn as fvnn
from typing import *

from seen2scene.tools.log_utils import get_logger

logger = get_logger(file_name=__file__, debug="Data")

@torch.no_grad()
def sparsify_grid(
    latent: fvnn.VDBTensor,
    sparity_ratio: float,
    ins_mask: Optional[torch.Tensor],
) -> Tuple[fvnn.VDBTensor, float]:
    """
    Keep **all** voxels where `band_mask == 1`, and randomly keep a `ratio` fraction
    of voxels where `band_mask == 0`.

    Notes:
    - `band_mask` is expected to be shape [N] aligned with `latent.data.jdata`.
    - `ratio` is scheduled linearly by epoch: 0 at epoch 0, 1 at `progressive_end`.
    - `object_bboxes` is the list of object bboxes in the field.
    """
    assert 0.0 <= sparity_ratio <= 1.0, "Sparity ratio must be between 0 and 1"
    device = latent.device
    keep_mask = torch.rand((len(latent.data.jdata),), device=device) < sparity_ratio

    inside_mask = ins_mask.any(dim=-1)  # [N_total]
    keep_mask |= inside_mask

    # Fix: Use runtime check instead of assertion (not disabled with -O flag)
    if not torch.any(keep_mask):
        raise RuntimeError("No voxels left after sparsity filtering")

    ijks = latent.grid.ijk.rmask(keep_mask)
    grid = fvdb.gridbatch_from_ijk(
        ijks, voxel_sizes=latent.grid.voxel_sizes, origins=latent.grid.origins
    )
    jagged = grid.jagged_like(latent.data.jdata[keep_mask])
    latent = fvnn.VDBTensor(grid, jagged)

    return latent, keep_mask


class Normalizer(nn.Module):
    def __init__(
        self,
        channels: int,
        momentum: float = 0.1,
        norm_by: Literal["minmax", "std"] = "minmax",
    ):
        super().__init__()
        self.momentum = momentum
        self.norm_by = norm_by
        self.register_buffer("running_mean", torch.zeros(channels))
        self.register_buffer("running_var", torch.ones(channels))
        self.register_buffer("running_min", torch.full((channels,), float("-1.0")))
        self.register_buffer("running_max", torch.full((channels,), float("1.0")))

    def stats_dict(self):
        return {
            "mean_val": self.running_mean.mean().item(),
            "std_val": self.running_var.sqrt().mean().item(),
            "min_val": self.running_min.min().item(),
            "max_val": self.running_max.max().item(),
        }

    @torch.no_grad()
    def track(self, value: torch.Tensor) -> None:
        dims = list(range(value.ndim - 1))  # exclude feature dimension
        batch_mean = value.mean(dim=dims)  # [C]
        batch_var = value.var(dim=dims, unbiased=False)
        batch_min = value.amin(dim=dims)
        batch_max = value.amax(dim=dims)

        # Update running stats
        self.running_mean = (
            1 - self.momentum
        ) * self.running_mean + self.momentum * batch_mean
        self.running_var = (
            1 - self.momentum
        ) * self.running_var + self.momentum * batch_var
        self.running_min = torch.min(self.running_min, batch_min)
        self.running_max = torch.max(self.running_max, batch_max)

    @torch.no_grad()
    def normalize(self, x: torch.Tensor):
        """
        x: [..., C]
        """
        view_shape = [1] * (x.ndim - 1) + [-1]

        if self.norm_by == "minmax":
            running_min = self.running_min.view(view_shape)
            running_max = self.running_max.view(view_shape)
            # Fix: Add epsilon to prevent division by zero
            x = (x - running_min) / (running_max - running_min + 1e-8)
            x = x * 2.0 - 1.0
        elif self.norm_by == "std":
            running_mean = self.running_mean.view(view_shape)
            running_var = self.running_var.view(view_shape)
            # Fix: Add epsilon to prevent division by zero
            x = (x - running_mean) / (running_var.sqrt() + 1e-8)
        else:
            raise NotImplementedError()

        return x

    @torch.no_grad()
    def denormalize(self, x: torch.Tensor):
        """
        x: [..., C]
        """
        view_shape = [1] * (x.ndim - 1) + [-1]
        if self.norm_by == "minmax":
            running_min = self.running_min.view(view_shape)
            running_max = self.running_max.view(view_shape)
            x = (x + 1.0) / 2.0
            x = x * (running_max - running_min) + running_min
        elif self.norm_by == "std":
            running_mean = self.running_mean.view(view_shape)
            running_var = self.running_var.view(view_shape)
            x = x * running_var.sqrt() + running_mean
        else:
            raise NotImplementedError()

        return x
