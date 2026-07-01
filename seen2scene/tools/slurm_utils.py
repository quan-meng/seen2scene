from pathos.multiprocessing import ProcessingPool as Pool
from rich.progress import track
import submitit
import torch
import os
from typing import *
import dataclasses
import functools
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
from typing import Union

from .log_utils import get_logger


@dataclasses.dataclass
class Base:
    slurm_job_name: str = "training"  # Job name
    gpus_per_node: int = 0  # Number of GPUs per node
    tasks_per_node: int = 1  # Number of tasks per node
    nodes: int = 1  # Number of nodes
    mem_gb: int = 40  # Memory per node
    cpus_per_task: int = 4  # Number of CPUs per task
    slurm_mail_type: Optional[str] = None  # "ALL"
    timeout_min: int = 60 * 48  # 2 days in minutes
    stderr_to_stdout: bool = True  # Redirect stderr to stdout
    cluster: Optional[str] = None  # Use "local" to run jobs locally
    folder: Optional[str] = None  # log


@dataclasses.dataclass
class Slurm(Base):
    slurm_time: str = "4-00:00:00"  # 4 days
    slurm_partition: str = "rtx_a6000_submit"  # gputype_{interactive,submit}: gtx_1080, rtx_2080, rtx_3090, rtx_a6000, a100
    slurm_exclude: str = "sorona,"  # "node1,node2,node3" uncomment for exclude nodes
    # slurm_qos: str = "deadline"           # "deadline" uncomment for deadline queue


"""
+-------------------------------+-------------------------------+-----------------+---------------+-----------------------+-----------------------------+----------------+
| Architecture                  | Slurm Partition               | Number of nodes | CPUs per node | CPU Memory per node   | GPUs per node               | Memory per GPU |
+-------------------------------+-------------------------------+-----------------+---------------+-----------------------+-----------------------------+----------------+
| HGX H100 (BayernKI)           | lrz-hgx-h100-94x4             | 30              | 96            | 768 GB                | 4 × NVIDIA H100             | 94 GB HBM2     |
| HGX A100                      | lrz-hgx-a100-80x4             | 5               | 96            | 1 TB                  | 4 × NVIDIA A100             | 80 GB HBM2     |
| DGX A100                      | lrz-dgx-a100-80x8             | 4               | 252           | 2 TB                  | 8 × NVIDIA A100             | 80 GB HBM2     |
| DGX A100 (MIG)                | lrz-dgx-a100-40x8-mig         | 1               | 252           | 1 TB                  | 8 × NVIDIA A100 (MIG)       | 40 GB HBM2     |
| DGX-1 V100                    | lrz-dgx-1-v100x8              | 1               | 76            | 512 GB                | 8 × NVIDIA Tesla V100       | 16 GB HBM2     |
| DGX-1 P100                    | lrz-dgx-1-p100x8              | 1               | 76            | 512 GB                | 8 × NVIDIA Tesla P100       | 16 GB HBM2     |
| HPE Intel Skylake + P100      | lrz-hpe-p100x4                | 1               | 28            | 256 GB                | 4 × NVIDIA Tesla P100       | 16 GB HBM2     |
| V100 GPU Nodes                | lrz-v100x2 (default)          | 4               | 19            | 368 GB                | 2 × NVIDIA Tesla V100       | 16 GB HBM2     |
| CPU Nodes                     | lrz-cpu                       | 12              | 18/28/38/94   | min. 360 GB           | --                          | --             |
+-------------------------------+-------------------------------+-----------------+---------------+-----------------------+-----------------------------+----------------+

MCML – Dedicated Partitions (QoS: mcml, max runtime 2 days)

+-------------------------------+-------------------------------+-----------------+---------------+-----------------------+-----------------------------+----------------+
| Architecture                  | Slurm Partition               | Number of nodes | CPUs per node | Memory per node       | GPUs per node               | Memory per GPU |
+-------------------------------+-------------------------------+-----------------+---------------+-----------------------+-----------------------------+----------------+
| HGX H100 Architecture         | mcml-hgx-h100-94x4            | 21              | 96            | 768 GB                | 4 × NVIDIA H100             | 94 GB          |
| HGX A100 Architecture         | mcml-hgx-a100-80x4            | 21              | 96            | 1 TB                  | 4 × NVIDIA A100             | 80 GB          |
| DGX A100 Architecture         | mcml-dgx-a100-40x8            | 8               | 256           | 1 TB                  | 8 × NVIDIA A100             | 40 GB          |
| HGX A100 (MIG)                | mcml-hgx-a100-80x4-mig        | 3               | --            | --                    | 4 × NVIDIA A100 (MIG)       | sliced (MIG)   |
+-------------------------------+-------------------------------+-----------------+---------------+-----------------------+-----------------------------+----------------+
"""


@dataclasses.dataclass
class LRZ(Base):
    slurm_time: str = "2-00:00:00"  # 4 days
    slurm_partition: str = "mcml-dgx-a100-40x8"


def group_dicts(list_dict: List[Dict[str, Any]], chunk_size: int = 1):
    """
    Group a list of dictionaries into chunks of a specified size.
    """
    # Initialize an empty list to hold the new dictionaries
    grouped_list = []

    # Iterate through the list in chunks
    for i in range(0, len(list_dict), chunk_size):
        chunk = list_dict[i : i + chunk_size]
        grouped_dict = defaultdict(
            list
        )  # Create a new dictionary where each value is a list

        # Populate the grouped dictionary
        for d in chunk:
            for key, value in d.items():
                grouped_dict[key].append(value)

        # Convert defaultdict to a normal dict and add to the final list
        grouped_list.append(dict(grouped_dict))

    return grouped_list


def slurm_func_wrapper(
    func: callable,
    num_workers: int = 1,
    fn_kwargs_share: Optional[dict] = {},
):
    """
    A wrapper function to call a function with a list of arguments.
    """

    @functools.wraps(func)
    def wrapper(fn_kwargs_list: Optional[list[str, dict]] = []):
        return run_with_mp(
            func,
            fn_kwargs_list=fn_kwargs_list,
            num_workers=num_workers,
            fn_kwargs_share=fn_kwargs_share.copy(),
        )

    return wrapper


def mp_func_wrapper(fn: Callable, fn_kwargs_share):
    """
    A wrapper function to call a function with a dictionary of keyword arguments.
    """

    @functools.wraps(fn)
    def wrapper(fn_kwargs: Union[Dict[str, Any], List[Any], Any]):
        if isinstance(fn_kwargs, dict):
            return fn(**fn_kwargs, **fn_kwargs_share.copy())
        elif isinstance(fn_kwargs, list):
            return fn(*fn_kwargs, **fn_kwargs_share.copy())
        else:
            return fn(fn_kwargs, **fn_kwargs_share.copy())

    return wrapper


def run_with_mp(
    fn: Callable,
    fn_kwargs_list: Optional[Union[List[Dict[str, Any]], List[Any]]] = [],
    num_workers: int = 1,
    fn_kwargs_share: Optional[Dict[str, Any]] = {},
):
    """
    A wrapper to run a function in parallel using multiprocessing with a pool of workers,
    with a progress bar.
    """
    # Create a logger within this function
    logger = get_logger(file_name=__file__)

    results = []
    if num_workers > 1:
        with Pool(processes=num_workers) as pool:
            fn_wrapped = mp_func_wrapper(fn, fn_kwargs_share=fn_kwargs_share)

            try:
                for result in pool.imap(fn_wrapped, fn_kwargs_list):
                    results.append(result)
            except Exception as e:
                logger.error(f"An error occurred: {e}")
                raise
    else:
        # Run the function sequentially if only one worker is specified
        for fn_kwargs in fn_kwargs_list:
            if isinstance(fn_kwargs, dict):
                result = fn(**fn_kwargs, **fn_kwargs_share)
            elif isinstance(fn_kwargs, list):
                result = fn(*fn_kwargs, **fn_kwargs_share)
            else:
                result = fn(fn_kwargs, **fn_kwargs_share)
            results.append(result)

    return results


def submit_jobs(
    fn: callable,
    fn_kwargs_list: Optional[list[str, dict]] = [],
    fn_kwargs_share: Optional[dict] = {},
    slurm_kwargs: dict = {},
    num_workers: int = 1,
    folder: Optional[str] = None,  # Folder to save the log
    num_groups: int = 1,
):
    """
    Submit a job to the cluster using the `submitit` library.

    Parameters:
    - fn: The function to run.
    - slurm_kwargs: A dictionary of SLURM parameters for the job.
    - fn_kwargs: Additional keyword arguments to pass to `fn`.
    - fn_kwargs_list: A list of parameters to distribute across the cluster. If None, the function will be run once.
    - wait_done: If True, wait for the job to finish before returning.
    - num_workers: The number of worker processes to use.

    Returns:
    - None
    """
    # Create a logger within this function
    logger = get_logger(file_name=__file__)

    if torch.cuda.is_available():
        logger.critical("CUDA is available, run the job locally")
        slurm_kwargs["cluster"] = "local"
    else:
        logger.critical("CUDA is not available, run the job on the cluster")

    if folder is None:
        folder = os.path.join("./log", slurm_kwargs["slurm_job_name"])
    os.makedirs(folder, exist_ok=True)

    cluster = slurm_kwargs.pop("cluster")
    slurm_nodes = slurm_kwargs.pop("nodes")

    if cluster == "local":
        executor = ThreadPoolExecutor(max_workers=num_workers)
    else:
        executor = submitit.AutoExecutor(folder=folder, cluster=cluster)
        executor.update_parameters(nodes=1, **slurm_kwargs)

    if num_groups > 1:
        assert (
            len(fn_kwargs_list) > 1
        ), "fn_kwargs_list must be provided when chunk_size > 1"
        fn_kwargs_list = group_dicts(fn_kwargs_list, chunk_size=num_groups)

    jobs = []
    if len(fn_kwargs_list) == 0:
        jobs = [executor.submit(fn, **fn_kwargs_share)]
    else:
        num_jobs = len(fn_kwargs_list)
        chunk = num_jobs // min(num_jobs, slurm_nodes)

        func = slurm_func_wrapper(
            fn, num_workers=num_workers, fn_kwargs_share=fn_kwargs_share
        )

        for i in track(
            range(0, num_jobs, chunk), description="Submitting jobs to cluster"
        ):
            jobs.append(
                executor.submit(func, fn_kwargs_list=fn_kwargs_list[i : i + chunk])
            )

    logger.info(f"Submitted {len(jobs)} jobs")

    if cluster == "local":
        [job.result() for job in jobs]

    return jobs
