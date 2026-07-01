import torch
import fvdb
import fvdb.nn as fvnn
from typing import List, Tuple, Literal


def coords_rotz90(ijks: torch.Tensor, k: Literal[1, 2, 3]):
    """
    Rotate coordinates around Z-axis by k×90 degrees (clockwise from above).

    Args:
        ijks: Coordinates of shape [..., 3] in world space (centered at origin)
        k: Number of 90° rotations (1, 2, or 3)

    Returns:
        Rotated coordinates

    Note:
        Uses continuous rotation matrices. For world coordinates where voxel
        centers are at positions (i - N/2 + 0.5) × voxel_size, this is
        mathematically equivalent to torch.rot90 on discrete voxel grids.

        Rotation matrices (Z-axis, clockwise from above):
        - k=1 (90°):  [[0, 1, 0], [-1, 0, 0], [0, 0, 1]]  →  (x,y,z) → (y,-x,z)
        - k=2 (180°): [[-1, 0, 0], [0, -1, 0], [0, 0, 1]] →  (x,y,z) → (-x,-y,z)
        - k=3 (270°): [[0, -1, 0], [1, 0, 0], [0, 0, 1]]  →  (x,y,z) → (-y,x,z)
    """
    device = ijks.device

    # Create rotation matrix based on the number of 90-degree rotations
    if k == 1:  # 90 degrees
        R = torch.tensor([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], device=device).float()
    elif k == 2:  # 180 degrees
        R = torch.tensor([[-1, 0, 0], [0, -1, 0], [0, 0, 1]], device=device).float()
    elif k == 3:  # 270 degrees
        R = torch.tensor([[0, -1, 0], [1, 0, 0], [0, 0, 1]], device=device).float()

    coords_new = ijks.flatten(0, ijks.ndim - 2)
    coords_new = torch.einsum("nj,ij->ni", coords_new, R)
    coords_new = coords_new.unflatten(0, ijks.shape[:-1])

    return coords_new


def coords_flip(ijks: torch.Tensor, dim: List[int]):
    """
    Flip coordinates along specified dimensions.

    Args:
        ijks: Coordinates of shape [..., 3] in world space (centered)
        dim: List of dimensions to flip [0=X, 1=Y, 2=Z]

    Returns:
        Flipped coordinates (negated along specified dimensions)

    Note:
        For world coordinates centered at origin, flipping is simply negation.
        This is mathematically equivalent to torch.flip for discrete grids when
        properly accounting for voxel centers at half-integer positions.
    """
    ijks = ijks.clone()
    for d in dim:
        ijks[..., d] = -1.0 * ijks[..., d]
    return ijks


def volume_rotz90(x: torch.Tensor, k: Literal[1, 2, 3]):
    """
    x: [B, C, G, G, G]
    k: int, 1, 2, 3
    """
    return torch.rot90(x, k=k, dims=[-2, -3])


def volume_flip(x: torch.Tensor, dims: List[int]):
    """
    x: [B, C, G, G, G]
    dims: List[int], -2, -3
    """
    return torch.flip(x, dims=dims)


def fvdb_flip(x: fvnn.VDBTensor, dims: List[int], voxel_bound: Tuple[int, int, int]):
    device = x.device
    voxel_bound = torch.tensor(voxel_bound, device=device, dtype=torch.int32)[
        None
    ]  # [1, 3]

    ijks_new = x.grid.ijk.jdata.clone().to(torch.int32)
    ijks_new[:, dims] = voxel_bound[:, dims] - 1 - ijks_new[:, dims]

    grid = fvdb.gridbatch_from_ijk(
        x.grid.jagged_like(ijks_new),
        voxel_sizes=x.grid.voxel_sizes,
        origins=x.grid.origins,
    )

    indices = grid.ijk_to_inv_index(x.grid.jagged_like(ijks_new))
    jagged = x.data[indices]

    return fvnn.VDBTensor(grid, jagged)


def fvdb_rotz90(x: fvnn.VDBTensor, voxel_bound: Tuple[int, int, int], k: int):
    device = x.device
    voxel_bound = torch.tensor(voxel_bound, device=device, dtype=torch.int32)[
        None
    ]  # [1, 3]

    ijks = x.grid.ijk.jdata.clone().float()

    ijks_new = coords_rotz90(ijks, k).to(torch.int32)
    # Shift x, y to make bounds unchanged
    if k == 1:
        ijks_new[:, 1] += voxel_bound[:, 1] - 1
    elif k == 2:
        ijks_new[:, :2] += voxel_bound[:, :2] - 1
    elif k == 3:
        ijks_new[:, 0] += voxel_bound[:, 0] - 1

    grid = fvdb.gridbatch_from_ijk(
        x.grid.jagged_like(ijks_new),
        voxel_sizes=x.grid.voxel_sizes,
        origins=x.grid.origins,
    )
    indices = grid.ijk_to_inv_index(x.grid.jagged_like(ijks_new))
    jagged = x.data[indices]
    return fvnn.VDBTensor(grid, jagged)
