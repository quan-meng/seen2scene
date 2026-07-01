import os
import sys
import warnings
import tyro
import dataclasses
from typing import *

# Prevent dual-import of seen2scene submodules. When running `python main.py` from
# the seen2scene/ directory, Python adds it to sys.path[0], allowing e.g.
# `models.sparse.basic.SparseTensor` alongside `seen2scene.models.sparse.basic.SparseTensor`.
# These become distinct classes, breaking isinstance() checks at runtime.
_this_dir = os.path.dirname(os.path.abspath(__file__))
sys.path = [p for p in sys.path if os.path.abspath(p) != _this_dir]

from pytorch_lightning.trainer import Trainer
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor

# Suppress FutureWarning from PyTorch's checkpoint.py about deprecated autocast API
# This is from PyTorch internals, not our code, and will be fixed in future PyTorch versions
warnings.filterwarnings("ignore", message=".*torch.cpu.amp.autocast.*is deprecated.*")
warnings.filterwarnings("ignore", message=".*torch.cuda.amp.autocast.*is deprecated.*")

from seen2scene.configs.opt import VAE, Generator, Control
from seen2scene.tools.common_utils import float2str
from seen2scene.tools.slurm_utils import submit_jobs


def main(args: Union[VAE, Generator, Control]):
    import torch
    from seen2scene.tools.common_utils import get_obj_from_str
    from seen2scene.tools.log_utils import get_logger
    from seen2scene.tools.lightning import (
        DataModuleFromConfig,
        CustomProgressBar,
        ModelCheckpoint,
        rank_zero_only_context,
        launch_logger,
    )

    torch.multiprocessing.set_sharing_strategy("file_system")

    logger = get_logger(file_name=__file__)
    seed_everything(args.seed)

    ckpt_path = None
    if args.resume is not None:
        voxel_size = float2str(args.data.voxel_size)
        ckpt_path = os.path.join(
            args.resume, "checkpoint", f"vxl_{voxel_size}_last.ckpt"
        )

    # Logger --------------------------------------------------------------
    train_logger = launch_logger(
        save_dir=args.log_dir,
        job_name=args.job_name,
        project=args.name,
        task_name=args.task.name,
        logger="wandb" if any(s in args.task.name for s in ["train", "val"]) else "csv",
    )
    train_logger.log_hyperparams(dataclasses.asdict(args))

    # Callbacks --------------------------------------------------------------
    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        ModelCheckpoint(
            dirpath=os.path.join(args.log_dir, "checkpoint"),
            save_last=True,
            every_n_epochs=1,
        ),
        CustomProgressBar(timestamp=args.timestamp, task_name=args.task.name),
    ]

    # Trainer --------------------------------------------------------------
    trainer = Trainer(
        benchmark=args.benchmark,
        max_epochs=args.max_epochs,
        max_steps=args.max_steps,
        accelerator=args.accelerator,
        devices=-1,
        strategy=args.strategy,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        callbacks=callbacks,
        logger=train_logger,
        profiler=args.profiler,
        precision=args.precision,
        accumulate_grad_batches=args.accumulate_grad_batches,
        gradient_clip_val=args.gradient_clip_val,
        gradient_clip_algorithm=args.gradient_clip_algorithm,
        num_sanity_val_steps=args.num_sanity_val_steps,
        enable_checkpointing=False,
        log_every_n_steps=1,
    )

    # Dataset --------------------------------------------------------------
    if args.task.name.startswith("train_"):
        splits = ["train", "val"]
    elif args.task.name.startswith("val_"):
        splits = ["val"]
    else:
        splits = ["test"]

    # Limit dataset size at DataLoader level for inference tasks so data loading
    # doesn't waste time on samples beyond num_samples.
    num_samples = (
        getattr(args.task, "num_samples", None)
        if not args.task.name.startswith("train_")
        else None
    )

    datamodule = DataModuleFromConfig(
        data_config=args.data,
        num_workers=args.num_workers,
        pin_memory=True,
        use_worker_init_fn=True,
        batch_size=args.batch_size,
        train_data_keys=args.train_data_keys,
        val_data_keys=args.val_data_keys,
        test_data_keys=args.test_data_keys,
        val_split=args.val_split,
        test_split=args.test_split,
        splits=splits,
        is_distributed="ddp" in args.strategy,
        num_samples=num_samples,
    )

    # Model ---------------------------------------------------
    with trainer.init_module():
        model = get_obj_from_str(args.target)(args)

    with rank_zero_only_context():
        with open(os.path.join(args.log_dir, f"model.txt"), "w") as f:
            f.writelines(repr(model))

    # Training/Validation/Test ---------------------------------------------------
    if args.task.name.startswith("train_"):
        logger.info("Running training step")
        trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)
    elif args.task.name.startswith("val_"):
        logger.info("Running validation step")
        trainer.validate(model, datamodule=datamodule, ckpt_path=ckpt_path)
    else:
        logger.info("Running test step")
        trainer.test(model, datamodule=datamodule, ckpt_path=ckpt_path)


if __name__ == "__main__":
    opt = tyro.cli(VAE | Generator | Control, config=(tyro.conf.CascadeSubcommandArgs,))
    submit_jobs(
        fn=main,
        slurm_kwargs=dataclasses.asdict(opt.slurm),
        fn_kwargs_share={"args": opt},
        folder=opt.slurm_folder,
    )
