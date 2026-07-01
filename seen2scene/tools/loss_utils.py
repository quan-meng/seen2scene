import torch
import fvdb.nn as fvnn
import torch.nn.functional as F
from typing import *

from seen2scene.models.fields import TSDFPatch
from seen2scene.configs.dataset import Voxel
from seen2scene.tools.log_utils import get_logger

logger = get_logger(__name__, debug="loss_utils")


def masked_weighted_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    weighting: Optional[torch.Tensor] = None,
    mask: Optional[torch.Tensor] = None,
    loss_type: Literal["l1", "l2", "cos"] = "l2",
    batch_reduction: Optional[str] = "mean",
) -> torch.Tensor:
    """
    Args:
        pred: [B, ...]
        target: [B, ...]
        weighting: [B, ...]
        mask: [B, ...]
        loss_type: str
        batch_reduction: str
    """
    assert all(
        x == y for x, y in zip(pred.shape, target.shape)
    ), f"pred shape {pred.shape} != target shape {target.shape}"

    if loss_type == "l1":
        loss = F.l1_loss(pred, target, reduction="none")
    elif loss_type == "l2":
        loss = F.mse_loss(pred, target, reduction="none")
    elif loss_type == "cos":
        y = torch.ones((len(pred)), device=pred.device)
        loss = F.cosine_embedding_loss(
            input1=pred, input2=target, target=y, reduction="none"
        )

    mean_dims = [i for i in range(1, loss.ndim)]  # [1, 2, ...]

    weighting = 1.0 if weighting is None else weighting
    mask = torch.ones_like(loss) if mask is None else mask

    mask_sum = torch.clamp(mask.sum(dim=mean_dims), min=1e-6)
    loss = (weighting * loss * mask).sum(dim=mean_dims) / mask_sum

    if batch_reduction == "mean":
        return loss.mean()
    elif batch_reduction == "sum":
        return loss.sum()
    else:
        return loss  # no reduction


def cross_entropy(
    pd_struct: fvnn.VDBTensor, gt_struct: torch.Tensor, mask_unknown: bool = True
):
    device = gt_struct.device
    pd_logits = pd_struct.data.jdata  # [N, 3]

    # Fix: Use runtime check instead of assertion (not disabled with -O flag)
    max_indices = torch.max(pd_struct.grid.ijk.jdata, dim=0)[0]
    if torch.any(max_indices >= torch.tensor(gt_struct.shape[-3:], device=device)):
        raise IndexError(
            f"pd_struct grid index out of range! "
            f"Max indices: {max_indices.tolist()}, gt_struct shape: {gt_struct.shape[-3:]}"
        )

    gt_category = gt_struct[
        pd_struct.grid.jidx,
        0,
        pd_struct.grid.ijk.jdata[:, 0],
        pd_struct.grid.ijk.jdata[:, 1],
        pd_struct.grid.ijk.jdata[:, 2],
    ]
    if mask_unknown:
        known_mask = gt_category != Voxel.UNKNOWN
        if known_mask.sum() == 0:
            # Fix: Return zero that maintains gradient flow
            return (pd_logits * 0.0).sum()
        gt_category = gt_category[known_mask]
        pd_logits = pd_logits[known_mask]

    gt_category = torch.where(gt_category == Voxel.BAND, 1, 0)

    loss = F.binary_cross_entropy_with_logits(pd_logits[:, 0], gt_category.float())

    return loss


@torch.no_grad()
def struct_acc(pd_struct: fvnn.VDBTensor, gt_struct: torch.Tensor):
    pd_category = pd_struct.data.jdata[:, 0]
    gt_category = gt_struct[
        pd_struct.grid.jidx,
        0,
        pd_struct.grid.ijk.jdata[:, 0],
        pd_struct.grid.ijk.jdata[:, 1],
        pd_struct.grid.ijk.jdata[:, 2],
    ].long()

    known_mask = gt_category != Voxel.UNKNOWN
    if known_mask.sum() == 0:
        # Fix: Return zero that maintains gradient flow (though @torch.no_grad())
        return (pd_category * 0.0).sum()
    gt_category = gt_category[known_mask]
    pd_category = pd_category[known_mask]

    pd_category = torch.sigmoid(pd_category) > 0.5
    gt_category = torch.where(gt_category == Voxel.EMPTY, 0, 1)

    return torch.mean((pd_category == gt_category).float())


def structure_loss(
    structure_pred: Dict[int, fvnn.VDBTensor],
    struct_tree: Dict[int, torch.Tensor],
    mask_unknown: bool = True,
    compute_metric: bool = False,
    structure_weight: float = 1.0,
):
    loss_dict, metric_dict = OrderedDict(), OrderedDict()
    for feat_depth, pd_struct_i in structure_pred.items():
        gt_structure_i = struct_tree[feat_depth]

        if pd_struct_i.grid.total_voxels == 0:  # empty grid can cause error
            continue

        struct_loss_i = cross_entropy(
            pd_struct_i, gt_structure_i, mask_unknown=mask_unknown
        )

        loss_dict[f"struct-{feat_depth}"] = struct_loss_i * structure_weight

        if compute_metric:
            struct_acc_i = struct_acc(pd_struct_i, gt_structure_i)
            metric_dict |= {f"struct-acc-{feat_depth}": struct_acc_i}

    logger.debug(
        "metrics: " + ", ".join([f"{k}: {v:.4f}" for k, v in metric_dict.items()])
    )

    return loss_dict, metric_dict


def geometry_loss(
    pred: fvnn.VDBTensor,
    target: fvnn.VDBTensor,
    truncation: float,
    gaussian_tau: Optional[float] = None,
    geometry_weight: float = 1.0,
) -> Dict[str, torch.Tensor]:
    loss_dict = OrderedDict()

    # Fix: Use runtime checks instead of assertions (not disabled with -O flag)
    if not torch.allclose(pred.grid.origins, target.grid.origins):
        raise ValueError("pred and target grids have different origins")
    if not torch.allclose(pred.grid.voxel_sizes, target.grid.voxel_sizes):
        raise ValueError("pred and target grids have different voxel sizes")

    if pred.grid.total_voxels == 0 or target.grid.total_voxels == 0:
        # Fix: Return zero with gradient flow and early return
        loss_dict[f"geometry"] = (pred.data.jdata * 0.0).sum()
        return loss_dict

    pd_geometry = (
        target.grid.fill_from_grid(pred.data, pred.grid, truncation).jdata[:, 0]
        / truncation
    )
    gt_geometry = target.data.jdata[:, 0] / truncation
    if gaussian_tau is not None:
        weight = torch.exp(-(gt_geometry**2) / (2 * gaussian_tau**2))
    else:
        weight = 1.0

    loss = masked_weighted_loss(
        pd_geometry, gt_geometry, weighting=weight, loss_type="l1"
    )

    loss_dict[f"geometry"] = loss * geometry_weight

    return loss_dict


def sparse_rec_loss(
    pred: fvnn.VDBTensor,
    target: TSDFPatch,
    struct_feats: Dict[int, fvnn.VDBTensor],
    struct_tree: Dict[str, torch.Tensor],
    compute_metric: bool = False,
    mask_unknown: bool = True,
    structure_weight: float = 1.0,
    geometry_weight: float = 1.0,
    gaussian_tau: Optional[float] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    loss_dict = OrderedDict()

    struc_loss_dict, metric_dict = structure_loss(
        structure_pred=struct_feats,
        struct_tree=struct_tree,
        mask_unknown=mask_unknown,
        compute_metric=compute_metric,
        structure_weight=structure_weight,
    )
    loss_dict |= struc_loss_dict

    # Geometry Loss
    geo_loss_dict = geometry_loss(
        pred=pred,
        target=target.known_vdb(),
        truncation=target.truncation,
        gaussian_tau=gaussian_tau,
        geometry_weight=geometry_weight,
    )
    loss_dict |= geo_loss_dict

    return loss_dict, metric_dict
