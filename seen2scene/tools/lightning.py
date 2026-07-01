import numpy as np
import os
import wandb
import dataclasses
from typing import *
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Subset
from pytorch_lightning.callbacks import Callback, TQDMProgressBar
import torch.distributed as dist
from contextlib import contextmanager
from torch.utils.data import DistributedSampler

from .log_utils import get_logger
from seen2scene.tools.common_utils import (
    instantiate_from_config,
    float2str,
)
from seen2scene.dataset.common import custom_collate_fn, batch_to_Field

logger = get_logger(file_name=__file__, debug="lightning")


def worker_init_fn(worker_id):
    return np.random.seed(np.random.get_state()[1][0] + worker_id)


def str2float(s):
    try:
        return float(s)
    except ValueError:
        return s


def parse_profiler_summary(summary_text, top_n=20, sort_by="Mean duration (s)"):
    # Extract rows containing data (those with '|')
    rows = [
        [col.strip() for col in line.split("|")[1:-1]]
        for line in summary_text.split("\n")
        if "|" in line
    ]

    if not rows:
        return None

    # Separate header and data
    headers = rows[0]
    data_rows = rows[2:]

    # Convert numeric data to float
    numeric_rows = [[str2float(col) for col in row] for row in data_rows]

    # Sort by mean duration
    sort_by_idx = headers.index(sort_by)
    sorted_rows = sorted(numeric_rows, key=lambda x: x[sort_by_idx], reverse=True)

    # Get top 20 rows
    top_n_rows = sorted_rows[:top_n]

    return wandb.Table(columns=headers, data=top_n_rows)


@contextmanager
def rank_zero_only_context():
    if not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0:
        yield
    else:
        yield None


class DataModuleFromConfig(pl.LightningDataModule):
    def __init__(
        self,
        data_config: dataclasses.dataclass,
        num_workers: int,
        pin_memory: bool = True,
        use_worker_init_fn: bool = True,
        batch_size: Optional[Tuple[int, ...]] = None,
        splits: List[str] = ["train", "val", "test"],
        train_data_keys: Optional[List[str]] = None,
        val_data_keys: Optional[List[str]] = None,
        test_data_keys: Optional[List[str]] = None,
        val_split: str = "val",
        test_split: str = "test",
        is_distributed: bool = False,
        num_samples: Optional[int] = None,
    ):
        super().__init__()
        self.voxel_size = data_config.voxel_size
        self.patch_shape = data_config.patch_shape
        self.num_workers = min(os.cpu_count(), num_workers)
        self.pin_memory = pin_memory
        self.use_worker_init_fn = use_worker_init_fn
        self.is_distributed = is_distributed
        self.batch_size = batch_size
        self.splits = splits
        self.num_samples = num_samples

        self.datasets = {}
        for split in splits:
            data_cfg = dataclasses.asdict(data_config)
            if split == "train":
                data_cfg.update({"split": "train", "data_keys": train_data_keys})
            elif split == "val":
                data_cfg.update(
                    {
                        "split": val_split,
                        "data_keys": val_data_keys,
                        "augmentation": False,
                    }
                )
            elif split == "test":
                data_cfg.update(
                    {
                        "split": test_split,
                        "data_keys": test_data_keys,
                        "augmentation": False,
                    }
                )
            else:
                raise ValueError(f"Invalid split: {split}")
            self.datasets[split] = instantiate_from_config(data_cfg)

    def on_after_batch_transfer(self, batch, dataloader_idx) -> Any:
        return batch_to_Field(batch)

    def train_dataloader(self):
        sampler = (
            DistributedSampler(self.datasets["train"], shuffle=True)
            if self.is_distributed
            else None
        )
        init_fn = worker_init_fn if self.use_worker_init_fn else None
        return DataLoader(
            self.datasets["train"],
            batch_size=self.batch_size,
            sampler=sampler,
            shuffle=not self.is_distributed,
            worker_init_fn=init_fn,
            drop_last=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=2 if self.num_workers > 0 else None,
            persistent_workers=True,
            collate_fn=custom_collate_fn,
        )

    def val_dataloader(self):
        dataset = self.datasets["val"]
        if self.num_samples is not None:
            dataset = Subset(dataset, range(min(self.num_samples, len(dataset))))
        sampler = (
            DistributedSampler(dataset, shuffle=False) if self.is_distributed else None
        )
        init_fn = worker_init_fn if self.use_worker_init_fn else None
        return DataLoader(
            dataset,
            batch_size=min(4, self.batch_size),
            sampler=sampler,
            shuffle=not self.is_distributed,
            num_workers=self.num_workers,
            worker_init_fn=init_fn,
            pin_memory=self.pin_memory,
            collate_fn=custom_collate_fn,
        )

    def test_dataloader(self):
        dataset = self.datasets["test"]
        if self.num_samples is not None:
            dataset = Subset(dataset, range(min(self.num_samples, len(dataset))))
        sampler = (
            DistributedSampler(dataset, shuffle=False) if self.is_distributed else None
        )
        init_fn = worker_init_fn if self.use_worker_init_fn else None
        return DataLoader(
            dataset,
            batch_size=1,
            sampler=sampler,
            shuffle=False,
            worker_init_fn=init_fn,
            num_workers=0,
            pin_memory=self.pin_memory,
            collate_fn=custom_collate_fn,
        )


def launch_logger(
    save_dir: str,
    job_name: str = "",
    project: str = "sc_1st_stage",
    task_name: str = "",
    logger: Optional[str] = None,
):
    name = f"{job_name}_{task_name}"[:128]
    save_dir = os.path.join(save_dir, task_name)
    os.makedirs(save_dir, exist_ok=True)

    if logger == "wandb":
        from pytorch_lightning.loggers import WandbLogger

        return WandbLogger(
            name=name,
            project=project,
            save_dir=save_dir,
            id=name,
            version=name,
            log_model=False,
        )
    else:
        from pytorch_lightning.loggers import CSVLogger

        return CSVLogger(save_dir=save_dir, name=name, version=name)


class ModelCheckpoint(Callback):
    def __init__(
        self,
        dirpath: str,
        save_last: bool = True,
        every_n_epochs: int = 1,
        weights_only: bool = False,
    ):
        super().__init__()
        self.dirpath = dirpath
        with rank_zero_only_context():
            os.makedirs(self.dirpath, exist_ok=True)
        self.save_last = save_last
        self.every_n_epochs = every_n_epochs
        self.weights_only = weights_only

    def on_train_epoch_end(self, trainer, pl_module):
        voxel_size = trainer.datamodule.voxel_size
        voxel_size = float2str(voxel_size)

        if trainer.current_epoch % self.every_n_epochs == 0:
            checkpoint_path = os.path.join(self.dirpath, f"vxl_{voxel_size}_last.ckpt")
            trainer.save_checkpoint(checkpoint_path, weights_only=self.weights_only)


class CustomProgressBar(TQDMProgressBar):
    def __init__(
        self,
        timestamp: str,
        task_name: str,
        refresh_rate: int = 1,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(refresh_rate=refresh_rate, *args, **kwargs)
        self.task_name = task_name.upper()
        self.timestamp = timestamp

    def on_train_epoch_start(self, trainer, *args, **kwargs):
        super().on_train_epoch_start(trainer, *args, **kwargs)
        self.train_progress_bar.set_description(
            f"{self.timestamp} Epoch {trainer.current_epoch}"
        )

    def on_validation_epoch_start(self, trainer, *args, **kwargs):
        super().on_validation_epoch_start(trainer, *args, **kwargs)
        self.val_progress_bar.set_description(f"{self.timestamp} {self.task_name}")

    def on_test_epoch_start(self, trainer, *args, **kwargs):
        super().on_test_epoch_start(trainer, *args, **kwargs)
        self.test_progress_bar.set_description(f"{self.timestamp} {self.task_name}")
