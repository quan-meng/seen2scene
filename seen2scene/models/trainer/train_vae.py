import os
import torch
from typing import *

from seen2scene.configs.opt import VAE as ModelArgs
import seen2scene.tools.common_utils as cmt
from seen2scene.tools.log_utils import get_logger
from seen2scene.models.fields import TSDFPatch, MeshPatch
from seen2scene.models.trainer.common import Normalizer
from .base import Net as BaseTrainer
from seen2scene.tools import vis_utils

logger = get_logger(file_name=__file__, debug="trian_vae")


class Net(BaseTrainer):
    def __init__(self, args: ModelArgs):
        super().__init__(args)
        self.mask_unknown = args.mask_unknown

        self.norm = Normalizer(args.model.channels, momentum=args.bn_momentum)
        self.model = cmt.instantiate_from_config(args.model)

        if args.ckpt_path is not None:
            assert os.path.exists(
                args.ckpt_path
            ), f"Checkpoint {args.ckpt_path} not found"
            state_dict = torch.load(
                args.ckpt_path,
                map_location=lambda storage, loc: storage,
                weights_only=True,
            )["state_dict"]
            self.load_state_dict(state_dict, strict=False)
            logger.info(f"Resume VAE from {args.ckpt_path}")

    def losses(self, batch, return_recon: bool = False, compute_metric: bool = False):
        prefix = "train" if self.training else "val"

        loss_dict, state_dict, metric_dict, out_dict = self.model(
            batch[self.latent_key],
            sample_posterior=True,
            mask_unknown=self.mask_unknown,
            return_recon=return_recon,
        )
        self.norm.track(out_dict["latent"].data.jdata.detach())

        loss_dict = {f"{prefix}/{k}": v for k, v in loss_dict.items()}

        if compute_metric | return_recon:
            return loss_dict, state_dict, metric_dict, out_dict

        return loss_dict, state_dict

    def on_train_batch_end(self, outputs, batch, batch_idx):
        self.log_dict(
            self.norm.stats_dict(),
            sync_dist=True,
            rank_zero_only=True,
            batch_size=len(batch["scene_names"]),
        )

    @torch.no_grad()
    def render_outputs(self, batch, pred: TSDFPatch, **render_kwargs) -> OrderedDict:
        gen_dict = OrderedDict(
            {
                f"tsdf (Rec)": pred.render_field(**render_kwargs),
                f"structure (Rec)": pred.render_structure(**render_kwargs),
            }
        )
        tgt_dict = OrderedDict(
            {
                "GT": batch[self.latent_key].render_field(**render_kwargs),
                "GT (Structure)": batch[self.latent_key].render_structure(
                    **render_kwargs
                ),
            }
        )
        return gen_dict | tgt_dict

    @torch.no_grad()
    def export_out_fields(self, batch, pred: List[TSDFPatch]):
        scene_names = batch["scene_names"]

        fields_dict = {
            "tsdf (Target)": batch[self.latent_key],
            "tsdf (Ground Truth)": batch[self.latent_key],
            "tsdf (Rec)": pred,
        }

        for file_name, field_i in fields_dict.items():
            field_i.export_objects_as_json(self.log_dir, scene_names)
            field_i.export_objects_as_mesh(self.log_dir, scene_names)

            if isinstance(field_i, TSDFPatch) or isinstance(field_i, MeshPatch):
                meshes = field_i.export_mesh()
                for mesh, scene_name, bbox_i in zip(meshes, scene_names, field_i.bbox_world):
                    bbox_str = "_".join([f"{x:.01f}" for x in bbox_i[:3].cpu().numpy()])
                    out_dir_i = os.path.join(self.log_dir, f"{scene_name}_{bbox_str}")
                    os.makedirs(out_dir_i, exist_ok=True)
                    mesh.export(os.path.join(out_dir_i, f"{file_name}.ply"))

        if "mesh_gt" in batch:
            meshes = batch["mesh_gt"].export_mesh()
            for mesh, scene_name, bbox_i in zip(meshes, scene_names, batch["mesh_gt"].bbox_world):
                bbox_str = "_".join([f"{x:.01f}" for x in bbox_i[:3].cpu().numpy()])
                out_dir_i = os.path.join(self.log_dir, f"{scene_name}_{bbox_str}")
                os.makedirs(out_dir_i, exist_ok=True)
                if len(mesh.vertices) > 0:
                    mesh.export(os.path.join(out_dir_i, "mesh (Ground Truth).ply"))

            if "npz" in self.task.export_as and (isinstance(field_i, TSDFPatch)):
                field_i.export_scenes_as_npz(
                    self.log_dir, scene_names, postfix=file_name
                )

            if "bbox" in self.task.export_as:
                field_i.export_objects_as_mesh(
                    self.log_dir, scene_names, postfix=file_name
                )

        if "mesh_gt" in batch:
            meshes = batch["mesh_gt"].export_mesh()
            for mesh, scene_name, bbox_i in zip(meshes, scene_names, batch["mesh_gt"].bbox_world):
                bbox_str = "_".join([f"{x:.01f}" for x in bbox_i[:3].cpu().numpy()])
                out_dir_i = os.path.join(self.log_dir, f"{scene_name}_{bbox_str}")
                os.makedirs(out_dir_i, exist_ok=True)
                if len(mesh.vertices) > 0:
                    mesh.export(os.path.join(out_dir_i, "mesh (Ground Truth).ply"))

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        batch_size = len(batch["scene_names"])
        log_kwargs = {
            "prog_bar": False,
            "logger": True,
            "on_step": False,
            "on_epoch": True,
            "batch_size": batch_size,
            "sync_dist": True,
            "rank_zero_only": True,
        }

        loss_dict, state_dict, metric_dict, out_dict = self.losses(
            batch, return_recon=True, compute_metric=True
        )
        self.log_dict(loss_dict, **log_kwargs)
        self.log_dict(state_dict, **log_kwargs)
        self.log_dict(metric_dict, **log_kwargs)

        mesh_field_tgt = (
            batch["mesh_gt"] if "mesh_gt" in batch else batch[self.latent_key]
        )
        self.metrics["model"].update(
            field_pred=out_dict["recon_field"],
            volume_field_tgt=batch[self.latent_key],
            mesh_field_tgt=mesh_field_tgt,
        )
        if batch_idx == 0:
            images_dict = self.render_outputs(
                batch,
                out_dict["recon_field"],
                resolution=self.task.img_wh,
                num_views=1,
                backend=self.task.backend,
            )
            images, titles = vis_utils.batch2row_types2column(images_dict)
            image_grid = vis_utils.make_image_grid(images, titles)
            if self.trainer.is_global_zero:
                self.logger.log_image(
                    f"Validation_batch_{batch_idx}", [image_grid], step=self.global_step
                )

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        batch_size = len(batch["scene_names"])
        world_size = self.trainer.world_size
        rank = self.trainer.global_rank
        global_batch_idx = batch_idx * world_size + rank
        global_sample_idx = global_batch_idx * batch_size
        if global_sample_idx >= self.task.num_samples:
            return

        # Show scene name in progress bar
        scene_label = batch["scene_names"][0]
        pbar = getattr(self.trainer, "progress_bar_callback", None)
        if pbar is not None and hasattr(pbar, "test_progress_bar"):
            pbar.test_progress_bar.set_postfix_str(scene_label, refresh=True)

        gt_field = batch[self.latent_key]
        loss_dict, state_dict, metric_dict, outputs = self.model(
            gt_field,
            sample_posterior=True,
            mask_unknown=self.mask_unknown,
            return_recon=True,
        )
        preds = outputs["recon_field"]
        self.export_out_fields(batch, preds)
