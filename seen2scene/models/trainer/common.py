import math
import torch
import fvdb
import torch.nn as nn
import fvdb.nn as fvnn
from typing import *

from seen2scene.tools.log_utils import get_logger
from seen2scene.models import sparse as sp

logger = get_logger(file_name=__file__, debug="Data")


def cosine_schedule(step: int, total_steps: int) -> float:
    """Cosine schedule from 0.0 to 1.0, slow start then fast end (ease-in)."""
    t = step / max(total_steps, 1)
    return 1.0 - math.cos(math.pi / 2 * t)


@torch.no_grad()
def get_instance_mask(
    xyzs_list: List[torch.Tensor], object_bboxes: List[torch.Tensor]
) -> torch.Tensor:
    """
    Compute instance mask for each voxel given object bounding boxes.

    Notes:
        - `xyzs` is expected to be shape [N, 3], the world coordinates of each voxel center.
        - `object_bboxes` is expected to be a list of tensors, each of shape [M, 2, 3], representing the min and max corners of M bounding boxes for each object.
    """
    assert len(xyzs_list) == len(object_bboxes), "Length of xyzs_list and object_bboxes must match"
    device = xyzs_list[0].device

    bboxes_list = list(object_bboxes)
    xyzs_all = torch.cat(xyzs_list, dim=0)  # [N_total, 3]
    bboxes_all = torch.cat(bboxes_list, dim=0)  # [M_total, 2, 3]

    voxel_scene_ids = torch.repeat_interleave(
        torch.arange(len(xyzs_list), device=device),
        torch.tensor([x.shape[0] for x in xyzs_list], device=device),
    )
    bbox_scene_ids = torch.repeat_interleave(
        torch.arange(len(bboxes_list), device=device),
        torch.tensor([b.shape[0] for b in bboxes_list], device=device),
    )

    xyzs_all = xyzs_all[:, None, :]  # [N_total, 1, 3]
    bboxes_all = bboxes_all[None, :, :, :]  # [1, M_total, 2, 3]
    mask_nm = (xyzs_all >= bboxes_all[:, :, 0, :]) & (
        xyzs_all <= bboxes_all[:, :, 1, :]
    )
    mask_nm = mask_nm.all(dim=-1)  # [N_total, M_total]
    same_scene = voxel_scene_ids[:, None] == bbox_scene_ids[None, :]
    ins_mask = mask_nm & same_scene

    return ins_mask  # [N_total, M_total]


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

    # Ensure at least one voxel survives per batch element
    batch_size = len(latent.grid)
    jidx = latent.grid.ijk.jidx.long()  # [N] batch index per voxel
    kept_count = torch.bincount(jidx[keep_mask], minlength=batch_size)
    empty_batches = kept_count == 0  # [batch_size]
    if empty_batches.any():
        in_empty = empty_batches[jidx]  # [N] True for voxels in empty batches
        rand_val = torch.rand(len(jidx), device=device)
        rand_val[~in_empty] = -1.0
        max_rand = torch.full((batch_size,), -1.0, device=device)
        max_rand.scatter_reduce_(0, jidx, rand_val, reduce="amax")
        keep_mask |= (rand_val == max_rand[jidx]) & in_empty

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
