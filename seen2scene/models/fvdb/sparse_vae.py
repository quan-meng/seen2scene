import torch
import fvdb.nn as fvnn
import torch.nn as nn
import math
from typing import *
import torch.nn.functional as F

from seen2scene.models.fields import TSDFPatch
import seen2scene.tools.common_utils as tcu
from seen2scene.tools.log_utils import get_logger
from seen2scene.tools.loss_utils import sparse_rec_loss
from seen2scene.models.fields import MeshPatch

logger = get_logger(file_name=__file__, debug="geo_model")


class SparseVAE(nn.Module):
    def __init__(
        self,
        factor: int = 1,
        channels: Optional[int] = 1,
        encoder: str = "models.fvdb.encoder.Encoder",
        decoder: str = "models.fvdb.decoder.Decoder",
        structure_weight: float = 1.0,
        geometry_weight: float = 1.0,
        gaussian_tau: Optional[float] = None,
        feed_gt_structure: bool = False,
        kl_weight: float = 1.0e-3,
        **kwargs,
    ):
        super().__init__()
        self.channels = channels
        self.factor = factor
        self.structure_weight = structure_weight
        self.geometry_weight = geometry_weight
        self.gaussian_tau = gaussian_tau
        self.feed_gt_structure = feed_gt_structure
        self.kl_weight = kl_weight

        encoder_kwargs = {
            **kwargs,
            "in_channels": 1,
            "out_channels": self.channels,
            "factor": self.factor,
            "target": encoder,
        }
        self.encoder = tcu.instantiate_from_config(encoder_kwargs)

        decoder_kwargs = {
            "in_channels": self.channels,
            "factor": self.factor,
            "target": decoder,
            **kwargs,
        }
        self.decoder = tcu.instantiate_from_config(decoder_kwargs)

    def encode(
        self,
        x: TSDFPatch,
        sample_posterior: bool = True,
        mask_unknown: bool = True,
        keep_region: Optional[List[str]] = None,
    ) -> Tuple[fvnn.VDBTensor, dict, dict, dict]:
        loss_dict, state_dict = OrderedDict(), OrderedDict()

        values = x.band.data.jdata / x.truncation
        band = fvnn.VDBTensor(x.band.grid, x.band.data.jagged_like(values))

        latent, band_mask, encoder_stat_dict = self.encoder(
            x=band,
            structure=x.structure,
            mask_unknown=mask_unknown,
            keep_region=keep_region,
            object_bboxes=x.object_bboxes,
        )
        state_dict |= encoder_stat_dict

        mean, logvar = torch.chunk(latent.data.jdata, 2, dim=1)
        mean = mean.contiguous()
        logvar = logvar.contiguous()

        kl_loss = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())
        loss_dict |= {"loss_kl": kl_loss * self.kl_weight}

        if sample_posterior:
            std = torch.exp(0.5 * logvar)
            z = mean + std * torch.randn_like(std)
        else:
            z = mean

        latent = fvnn.VDBTensor(latent.grid, latent.grid.jagged_like(z))

        return latent, band_mask, loss_dict, state_dict

    def decode(
        self,
        latent: fvnn.VDBTensor,
        field_params: Optional[Dict[str, Any]] = None,
        unstable_cutoff: bool = False,
        return_recon: bool = False,
        decision_tree: Optional[fvnn.VDBTensor] = None,
    ) -> Dict[str, Any]:
        decoder_outs, state_dict = self.decoder(
            latent,
            truncation=field_params["truncation"],
            unstable_cutoff=unstable_cutoff,
            decision_tree=decision_tree,
        )

        if return_recon:
            field_params = field_params.copy()
            field_params["voxel_size"] = (
                decoder_outs["geometry"].voxel_sizes[0][0].item()
            )
            decoder_outs["recon_field"] = TSDFPatch(
                band=decoder_outs["geometry"], kwargs=field_params
            )

        return decoder_outs, state_dict

    def forward(
        self,
        target: TSDFPatch,
        sample_posterior: bool = True,
        unstable_cutoff: bool = False,
        return_recon: bool = False,
        compute_metric: bool = False,
        mask_unknown: bool = True,
        keep_region: Optional[List[str]] = None,
    ):
        loss_dict, state_dict, metric_dict, out_dict = (
            OrderedDict(),
            OrderedDict(),
            OrderedDict(),
            OrderedDict(),
        )

        feat_depth = int(math.log2(self.factor)) + 1

        latent, _, loss_dict_, state_dict_ = self.encode(
            target,
            sample_posterior=sample_posterior,
            mask_unknown=mask_unknown,
            keep_region=keep_region,
        )
        out_dict |= {"latent": latent}
        loss_dict |= loss_dict_
        state_dict |= state_dict_

        if self.feed_gt_structure:
            decision_tree = target.build_geometry_tree(
                tree_depth=int(math.log2(self.factor)) + 1
            )
        else:
            decision_tree = None

        decoder_outs, state_dict_ = self.decode(
            latent,
            field_params=target.kwargs,
            decision_tree=decision_tree,
            unstable_cutoff=unstable_cutoff,
            return_recon=return_recon,
        )
        state_dict |= state_dict_
        out_dict |= decoder_outs

        struct_tree = target.build_structure_tree(tree_depth=feat_depth)
        loss_dict_, metric_dict_ = sparse_rec_loss(
            pred=decoder_outs.pop("geometry"),
            target=target,
            struct_feats=decoder_outs["struct_feats"],
            struct_tree=struct_tree,
            mask_unknown=mask_unknown,
            structure_weight=self.structure_weight,
            geometry_weight=self.geometry_weight,
            gaussian_tau=self.gaussian_tau,
            compute_metric=compute_metric,
        )
        loss_dict |= loss_dict_
        metric_dict |= metric_dict_

        return loss_dict, state_dict, metric_dict, out_dict
