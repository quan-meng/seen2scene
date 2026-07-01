import copy
import hashlib

import torch
import torch.nn as nn
from typing import *

import seen2scene.tools.common_utils as cmt
from seen2scene.tools.log_utils import get_logger

logger = get_logger(file_name=__file__, debug="flow_matching")


def zero_module(module):
    """Zero out the parameters of a module and return it."""
    for p in module.parameters():
        p.detach().zero_()
    return module


def zero_bottleneck(dim, rank):
    """Low-rank zero projection: dim → rank → dim.  Only up-proj is zeroed."""
    return nn.Sequential(
        nn.Linear(dim, rank, bias=False),
        zero_module(nn.Linear(rank, dim, bias=False)),
    )


class ControlBranch(nn.Module):
    """SparseDiT branch with zero convolutions, following the original ControlNet.

    Input side:  branch receives x_t + zero_proj_in(condition).
                 At init zero_proj_in outputs 0, so branch sees pure x_t.
    Output side: each block intermediate goes through a low-rank zero projection.
                 At init all controls are exactly zero → base model unchanged.
    """

    def __init__(
        self,
        model: nn.Module,
        num_blocks: int,
        in_channels: int,
        model_channels: int,
        zero_proj_rank: int = 64,
    ):
        super().__init__()
        self.model = model
        # Input zero conv: projects condition into x_t space (starts at zero)
        self.input_zero_proj = zero_module(nn.Linear(in_channels, in_channels))
        # Output zero convs: low-rank bottleneck per block (starts at zero)
        self.zero_projs = nn.ModuleDict(
            {
                str(i): zero_bottleneck(model_channels, zero_proj_rank)
                for i in range(num_blocks)
            }
        )

    def forward(self, condition, t, xt=None, **kwargs):
        """
        Args:
            condition: SparseTensor — source latent (latent_src), same grid as xt.
            t: timesteps.
            xt: SparseTensor — noised target latent (model_input), same grid as condition.
            **kwargs: embed, return_intermediates, etc. passed to the branch model.
        """
        # Combine x_t + zero_proj(condition) as branch input (like original ControlNet)
        if xt is not None:
            cond_feats = self.input_zero_proj(condition.feats)
            branch_input = xt.replace(xt.feats + cond_feats)
        else:
            branch_input = condition

        kwargs["return_intermediates"] = True
        intermediates = self.model(branch_input, t, **kwargs)
        return {
            k: v.replace(self.zero_projs[k](v.feats)) for k, v in intermediates.items()
        }


class Net(nn.Module):
    def __init__(
        self,
        gen_cfg: Dict[str, Any] = {},
        gen_ckpt: Optional[str] = None,
        zero_proj_rank: int = 64,
        **kwargs,
    ):
        super().__init__()

        self.base = {"model": cmt.instantiate_from_config(gen_cfg)}
        self.base["model"].requires_grad_(False)
        self.base["model"].eval()

        state_dict = torch.load(
            gen_ckpt, map_location=lambda storage, loc: storage, weights_only=True
        )["state_dict"]
        state_dict = {
            k[len("model.") :]: v
            for k, v in state_dict.items()
            if k.startswith("model.")
        }
        self.base["model"].load_state_dict(state_dict)
        self._gen_ckpt_path = gen_ckpt
        self._gen_ckpt_sha = self._file_sha(gen_ckpt)
        logger.info(f"Resume Generator from {gen_ckpt}")

        # Branch: deepcopy of the pretrained base model (like original ControlNet)
        branch_model = copy.deepcopy(self.base["model"].model)
        model_cfg = gen_cfg["model_cfg"]
        logger.info("Initialized branch via deepcopy of pretrained base model")

        # Wrap with zero convolutions at input and every output
        self.branch = ControlBranch(
            branch_model,
            num_blocks=model_cfg["num_blocks"],
            in_channels=model_cfg["in_channels"],
            model_channels=model_cfg["model_channels"],
            zero_proj_rank=zero_proj_rank,
        )

    @staticmethod
    def _file_sha(path: str) -> str:
        """Compute SHA-256 of a file."""
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()

    def maybe_reload_base(self) -> bool:
        """Reload the frozen base model if its checkpoint file has changed.

        Returns True if reloaded, False otherwise.
        """
        sha = self._file_sha(self._gen_ckpt_path)
        if sha == self._gen_ckpt_sha:
            return False
        state_dict = torch.load(
            self._gen_ckpt_path,
            map_location=lambda storage, loc: storage,
            weights_only=True,
        )["state_dict"]
        state_dict = {
            k[len("model."):]: v
            for k, v in state_dict.items()
            if k.startswith("model.")
        }
        self.base["model"].load_state_dict(state_dict)
        self.base["model"].requires_grad_(False)
        self.base["model"].eval()
        self._gen_ckpt_sha = sha
        logger.info(f"Reloaded base model from {self._gen_ckpt_path} (sha changed)")
        return True

    def to(self, *args, **kwargs):
        """Override to() to ensure base model is also moved."""
        self = super().to(*args, **kwargs)
        self.base["model"] = self.base["model"].to(*args, **kwargs)
        return self

    def cuda(self, device=None):
        """Override cuda() to ensure base model is also moved."""
        self = super().cuda(device)
        self.base["model"] = self.base["model"].cuda(device)
        return self

    def cpu(self):
        """Override cpu() to ensure base model is also moved."""
        self = super().cpu()
        self.base["model"] = self.base["model"].cpu()
        return self

    def losses(self, *args, **kwargs):
        return self.base["model"].losses(*args, controlnet=self.branch, **kwargs)

    def sample(self, *args, **kwargs):
        return self.base["model"].sample(*args, controlnet=self.branch, **kwargs)

    def multi_sample(self, *args, **kwargs):
        return self.base["model"].multi_sample(*args, controlnet=self.branch, **kwargs)
