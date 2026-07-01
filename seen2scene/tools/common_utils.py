import argparse
import io
import json
import logging
import os
import shlex
import socket
import subprocess
import warnings
import torch
import numpy as np
import random
import torch.nn as nn
import copy
import signal
from functools import wraps
from pathlib import Path
import importlib
from itertools import zip_longest
import torch.nn.functional as F
from typing import *
import dataclasses
from inspect import isfunction
import trimesh

logger = logging.getLogger(__name__)


def bbox2str(bbox):
    assert (
        len(bbox) == 6
    ), "Expected bbox to have 6 values [x_min, y_min, z_min, x_max, y_max, z_max]"
    if isinstance(bbox, torch.Tensor):
        bbox = bbox.cpu().numpy()
    return "_".join([f"{x:.01f}" for x in bbox[:3]])


def value2tuple(x: Union[int, float, List, Tuple], dims: int = 3) -> Union[List, Tuple]:
    """Convert a scalar value to a tuple by repeating it, or pass through tuples/lists.

    Args:
        x: Input value that is either a scalar (int/float) or already a list/tuple.
        dims: Number of dimensions to repeat the scalar value. Default is 3 for 3D operations.

    Returns:
        If x is already a list or tuple, returns x unchanged. If x is a scalar,
        returns a list with x repeated dims times.

    Examples:
        >>> value2tuple(5, dims=3)
        [5, 5, 5]
        >>> value2tuple([1, 2, 3], dims=3)
        [1, 2, 3]
    """
    return x if isinstance(x, (list, tuple)) else [x] * dims


def list_to_element(x: Union[Any, List[Any]]) -> Any:
    """Extract the first element from a list, or return the value if not a list.

    Args:
        x: Input value that may be a list or a single element.

    Returns:
        If x is a list, returns x[0]. Otherwise returns x unchanged.

    Examples:
        >>> list_to_element([42])
        42
        >>> list_to_element(42)
        42
    """
    if isinstance(x, list):
        return x[0]
    else:
        return x


def float2str(x: float) -> str:
    """Convert a float to a string with underscores replacing decimal points.

    Useful for creating filesystem-safe names from float values like voxel sizes.

    Args:
        x: Float value to convert.

    Returns:
        String representation with 3 decimal places and "." replaced by "_".

    Examples:
        >>> float2str(0.011)
        '0_011'
        >>> float2str(1.5)
        '1_500'
    """
    return f"{x:.3f}".replace(".", "_")


def ensure_list(data: Any) -> List[Any]:
    """Wrap a value in a list if it's not already a list.

    Args:
        data: Input value that may or may not be a list.

    Returns:
        If data is already a list, returns data unchanged. Otherwise returns [data].

    Examples:
        >>> ensure_list([1, 2, 3])
        [1, 2, 3]
        >>> ensure_list(42)
        [42]
    """
    return data if isinstance(data, list) else [data]


def setup_seed(seed: int) -> None:
    """Set random seeds for reproducibility across all random number generators.

    Configures seeds for PyTorch (CPU and CUDA), NumPy, and Python's random module
    to ensure deterministic behavior across runs.

    Args:
        seed: Integer seed value to use for all RNG initialization.

    Notes:
        Also sets torch.backends.cudnn.deterministic = True to ensure deterministic
        CUDNN operations, which may reduce performance slightly.

    Examples:
        >>> setup_seed(42)
        # All subsequent random operations will be deterministic
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def is_array_all_integer(arr: np.ndarray) -> bool:
    """Check if all elements in a NumPy array are integers (or very close to integers).

    Args:
        arr: NumPy array to check.

    Returns:
        True if all array elements are within floating-point tolerance of integers.
        Uses np.isclose to handle floating-point precision issues.

    Examples:
        >>> is_array_all_integer(np.array([1.0, 2.0, 3.0]))
        True
        >>> is_array_all_integer(np.array([1.1, 2.2, 3.3]))
        False
    """
    return np.all(np.isclose(arr, np.round(arr)))


def sample_data(loader: Iterable, loop: bool = False) -> Iterator:
    """Create a generator that yields batches from a data loader.

    Args:
        loader: Iterable data loader (e.g., PyTorch DataLoader) to sample from.
        loop: If True, infinitely loop through the data loader. If False, iterate
            once and stop.

    Yields:
        Batches from the data loader.

    Examples:
        >>> for batch in sample_data(train_loader, loop=False):
        ...     process(batch)

        >>> # Infinite sampling for training
        >>> sampler = sample_data(train_loader, loop=True)
        >>> batch = next(sampler)
    """
    if loop:
        while True:
            for batch in loader:
                yield batch
    else:
        for batch in loader:
            yield batch


def get_percentage(data_key: str) -> float:
    """Extract percentage value from a data key string.

    Parses data keys formatted like "prefix_p_0.5" to extract the percentage value.

    Args:
        data_key: String containing "_p_" followed by a percentage value.

    Returns:
        Float percentage value extracted from the key.

    Raises:
        ValueError: If "p" is not found in the splits.

    Examples:
        >>> get_percentage("partial_p_0.3")
        0.3
    """
    splits = data_key.split("_")
    index = splits.index("p")
    return float(splits[index + 1])


def exists(x: Any) -> bool:
    """Check if a value is not None.

    Args:
        x: Value to check.

    Returns:
        True if x is not None, False otherwise.

    Examples:
        >>> exists(42)
        True
        >>> exists(None)
        False
    """
    return x is not None


def default(val: Any, d: Union[Any, Callable]) -> Any:
    """Return value if it exists, otherwise return a default.

    Args:
        val: Value to check and potentially return.
        d: Default value or callable that returns a default value.

    Returns:
        val if val is not None, otherwise d() if d is callable, or d itself.

    Examples:
        >>> default(42, 0)
        42
        >>> default(None, 0)
        0
        >>> default(None, lambda: expensive_computation())
        # Only calls expensive_computation if needed
    """
    if exists(val):
        return val
    return d() if isfunction(d) else d


def count_params(model: nn.Module, verbose: bool = False) -> int:
    """Count the total number of parameters in a PyTorch model.

    Args:
        model: PyTorch model to count parameters for.
        verbose: If True, print the parameter count to stdout.

    Returns:
        Total number of parameters in the model.

    Examples:
        >>> model = nn.Linear(10, 5)
        >>> count_params(model, verbose=True)
        Linear has 0.06 M params.
        55
    """
    total_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"{model.__class__.__name__} has {total_params * 1.e-6:.2f} M params.")
    return total_params


def get_obj_from_str(string: str, reload: bool = False) -> Any:
    """Dynamically import and return a class or function from a module string.

    Args:
        string: Fully qualified import path like "module.submodule.ClassName".
        reload: If True, reload the module before getting the object. Useful
            for development when module code changes.

    Returns:
        The class or function object referenced by the string.

    Raises:
        ModuleNotFoundError: If the module doesn't exist.
        AttributeError: If the class/function doesn't exist in the module.

    Examples:
        >>> cls = get_obj_from_str("torch.nn.Linear")
        >>> model = cls(10, 5)
    """
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    try:
        return getattr(importlib.import_module(module, package=None), cls)
    except ModuleNotFoundError:
        # Saved configs from before package rename may use non-prefixed paths
        # (e.g. "models.trainer.train_gen.Net" instead of "seen2scene.models.trainer.train_gen.Net")
        if not module.startswith("seen2scene."):
            return getattr(importlib.import_module(f"seen2scene.{module}", package=None), cls)
        raise


def instantiate_from_config(cfg: Union[Dict, dataclasses.dataclass]) -> Any:
    """Instantiate an object from a configuration dictionary or dataclass.

    Expects configuration with a "target" key specifying the class import path,
    and remaining keys as initialization arguments. The "name" key is ignored
    if present.

    Args:
        cfg: Configuration as dictionary or dataclass. Must contain "target" key
            with fully qualified class path.

    Returns:
        Instantiated object of the type specified by cfg["target"], initialized
        with the remaining configuration parameters.

    Examples:
        >>> config = {"target": "torch.nn.Linear", "in_features": 10, "out_features": 5}
        >>> model = instantiate_from_config(config)
        >>> isinstance(model, torch.nn.Linear)
        True

        >>> from dataclasses import dataclass
        >>> @dataclass
        ... class ModelCfg:
        ...     target: str = "torch.nn.Linear"
        ...     in_features: int = 10
        ...     out_features: int = 5
        >>> model = instantiate_from_config(ModelCfg())
    """
    if dataclasses.is_dataclass(cfg):
        cfg = dataclasses.asdict(cfg)

    cfg = copy.deepcopy(cfg)
    target = cfg.pop("target")
    if "name" in cfg:
        cfg.pop("name")

    return get_obj_from_str(target)(**cfg)


def dict_to_namespace(config_dict: Dict[str, Any]) -> argparse.Namespace:
    """Recursively convert a dictionary to an argparse.Namespace object.

    Nested dictionaries are recursively converted to nested Namespace objects.

    Args:
        config_dict: Dictionary to convert, potentially with nested dictionaries.

    Returns:
        argparse.Namespace with dict keys as attributes. Nested dicts become
        nested Namespaces.

    Examples:
        >>> config = {"lr": 0.001, "model": {"layers": 4}}
        >>> ns = dict_to_namespace(config)
        >>> ns.lr
        0.001
        >>> ns.model.layers
        4
    """
    namespace = argparse.Namespace()
    for key, value in config_dict.items():
        if isinstance(value, dict):
            new_value = dict_to_namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def namespace_to_dict(
    namespace: Union[argparse.Namespace, Dict, Any],
) -> Union[Dict, Any]:
    """Recursively convert an argparse.Namespace object to a dictionary.

    Also handles nested Namespaces and dictionaries, recursively converting all
    Namespace objects to dictionaries.

    Args:
        namespace: Namespace, dict, or other value to convert.

    Returns:
        If input is a Namespace, returns a dict with Namespace attributes as keys.
        Nested Namespaces are recursively converted. If input is a dict, processes
        values recursively. Otherwise returns input unchanged.

    Examples:
        >>> ns = argparse.Namespace(lr=0.001, model=argparse.Namespace(layers=4))
        >>> namespace_to_dict(ns)
        {'lr': 0.001, 'model': {'layers': 4}}
    """
    if isinstance(namespace, argparse.Namespace):
        namespace_dict = vars(namespace)
        for key, value in namespace_dict.items():
            namespace_dict[key] = namespace_to_dict(value)
        return namespace_dict
    elif isinstance(namespace, dict):
        for key, value in namespace.items():
            namespace[key] = namespace_to_dict(value)
        return namespace
    else:
        return namespace


def recursive_to(
    a: Union[Dict, torch.Tensor, List, int, float, str, None],
    device: Union[str, torch.device],
) -> Union[Dict, torch.Tensor, List, int, float, str, None]:
    """Recursively move tensors in nested data structures to a specified device.

    Handles nested dictionaries, lists, and tensors. Primitive types (int, float,
    str, None) are returned unchanged.

    Args:
        a: Data structure potentially containing tensors. Can be dict, list, tensor,
            or primitive type.
        device: Target device (e.g., "cuda", "cpu", or torch.device object).

    Returns:
        Same structure as input with all tensors moved to the specified device.

    Raises:
        NotImplementedError: If the input type is not supported (not dict, list,
            tensor, or primitive).

    Examples:
        >>> data = {"x": torch.randn(3, 3), "y": [torch.randn(2, 2)]}
        >>> cuda_data = recursive_to(data, "cuda")

        >>> tensors = [torch.randn(10), torch.randn(20)]
        >>> cpu_tensors = recursive_to(tensors, "cpu")
    """
    if isinstance(a, dict):
        return {k: recursive_to(v, device) for k, v in a.items()}
    elif isinstance(a, torch.Tensor):
        return a.to(device)
    elif isinstance(a, list):
        return [recursive_to(v, device) for v in a]
    elif isinstance(a, int) or isinstance(a, float) or isinstance(a, str) or a is None:
        return a
    else:
        raise NotImplementedError


def uniform_3d_points(resolution=512, bbox=[-1, -1, -1, 1, 1, 1], device="cpu"):
    if isinstance(resolution, int):
        resolution = [resolution, resolution, resolution]
    x = torch.linspace(bbox[0], bbox[3], resolution[0], device=device)
    y = torch.linspace(bbox[1], bbox[4], resolution[1], device=device)
    z = torch.linspace(bbox[2], bbox[5], resolution[2], device=device)
    xyz = torch.stack(torch.meshgrid(x, y, z, indexing="ij"), -1)  # [G, G, G, 3]

    return xyz


def random_3d_points(num_points, bbox=[-1, -1, -1, 1, 1, 1], device="cpu"):
    if isinstance(num_points, int):
        num_points = [num_points, num_points, num_points]
    x = torch.rand(num_points[0], device=device) * (bbox[3] - bbox[0]) + bbox[0]
    y = torch.rand(num_points[1], device=device) * (bbox[4] - bbox[1]) + bbox[1]
    z = torch.rand(num_points[2], device=device) * (bbox[5] - bbox[2]) + bbox[2]
    xyz = torch.stack((x, y, z), -1)  # [P, 3]

    return xyz


def uniform_2d_points(resolution=512, bbox=[-1, -1, 1, 1], device="cpu"):
    w, h = torch.meshgrid(
        [
            torch.linspace(bbox[0], bbox[2], resolution),
            torch.linspace(bbox[1], bbox[3], resolution),
        ],
        indexing="ij",
    )
    coord = torch.stack((w, h), dim=-1)  # [G, G, 2]

    return coord


class FourierFeatureTransform(nn.Module):
    def __init__(self, num_input_channels, mapping_size, scale=1.0):
        super().__init__()

        self._num_input_channels = num_input_channels
        self._mapping_size = mapping_size
        self._B = nn.Parameter(
            torch.randn((num_input_channels, mapping_size)) * scale, requires_grad=False
        )
        self.out_channels = mapping_size * 2

    def forward(self, x):
        """
        Args:
            x: [B, C, G1, G2, G3]
        """
        # x = (x @ self._B)
        x = torch.einsum("bcijk,cd->bdijk", x, self._B)

        x = 2 * np.pi * x
        return torch.cat([torch.sin(x), torch.cos(x)], dim=1)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def get_model_size(model):
    num_params = count_parameters(model)
    model_size_mb = (
        num_params * 4 / (1024**2)
    )  # Convert bytes to megabytes (1 float32 = 4 bytes)

    return f"{model_size_mb:.2f} MB"


def rigid_transform(xyz, transform):
    """Applies a rigid transform to an (N, 3) pointcloud."""
    xyz_h = np.hstack([xyz, np.ones((len(xyz), 1), dtype=np.float32)])
    xyz_t_h = np.dot(transform, xyz_h.T).T
    return xyz_t_h[:, :3]


def get_view_frustum(img_hw, cam_intr, cam_pose, far=10.0):
    """Get corners of 3D camera view frustum of depth image"""
    im_h, im_w = img_hw
    view_frust_pts = np.array(
        [
            (np.array([0, 0, 0, im_w, im_w]) - cam_intr[0, 2])
            * np.array([0, far, far, far, far])
            / cam_intr[0, 0],
            (np.array([0, 0, im_h, 0, im_h]) - cam_intr[1, 2])
            * np.array([0, far, far, far, far])
            / cam_intr[1, 1],
            np.array([0, far, far, far, far]),
        ]
    )
    view_frust_pts = rigid_transform(view_frust_pts.T, cam_pose).T
    return view_frust_pts


def tsdf2known_mask(tsdf, known_ratio, truncation):
    return tsdf > (-1.0 * known_ratio * truncation)


def occupancy2known_mask(occupancy, **kwargs):
    return occupancy > 0.5


def tsdf2surface_mask(tsdf, known_ratio, truncation):
    return tsdf.abs() < known_ratio * truncation


def sample_random_pts(num_pts: torch.Tensor, volume: torch.Tensor):
    coords = torch.randn((num_pts, 3), device=volume.device)  # [N, 3]
    coords = coords * 2.0 - 1.0

    values = F.interpolate(volume, coords, mode="nearest")

    return coords, values


def timeout(seconds, error_message="Function call timed out"):
    """
    Decorator that adds a timeout to a function.

    Args:
        seconds (int): Maximum allowed time in seconds
        error_message (str): Message to display when timeout occurs

    Usage:
        @timeout(5)
        def slow_function():
            ...
    """

    def decorator(func):
        def _handle_timeout(signum, frame):
            raise TimeoutError(error_message)

        @wraps(func)
        def wrapper(*args, **kwargs):
            # Set the timeout handler
            signal.signal(signal.SIGALRM, _handle_timeout)
            signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
            finally:
                # Disable the alarm
                signal.alarm(0)
            return result

        return wrapper

    return decorator


def get_patch_shape(bbox_world: torch.Tensor, voxel_size: float) -> List[int]:
    bbox_world = bbox_world.cpu().numpy()
    dense_dims = (bbox_world[:, 3:] - bbox_world[:, :3]) / voxel_size
    dense_dims = np.round(dense_dims).astype(np.int32)  # [N, 3]
    assert np.all(
        dense_dims[[0]] == dense_dims
    ), f"dense_dims {dense_dims} must be the same"
    dense_dims = dense_dims[0].tolist()  # [3]

    return dense_dims


def interleave_lists(lists):
    return [x for gp in zip_longest(*lists) for x in gp if x is not None]


def lengths_to_layout(lengths: List[int]) -> List[slice]:
    offsets = [0] + np.cumsum(lengths).tolist()
    return [slice(offsets[i], offsets[i + 1]) for i in range(len(offsets) - 1)]


def clip_mesh(
    mesh: Union[str, trimesh.Trimesh],
    clip_bbox: Tuple[float, float, float, float, float, float],
    export_path: Optional[str] = None,
    cap: bool = False,
) -> trimesh.Trimesh:
    if isinstance(mesh, str):
        mesh = trimesh.load_mesh(mesh, process=False, maintain_order=True)

    # Create clipping box and perform clipping
    bbox_min, bbox_max = clip_bbox[:3], clip_bbox[3:]
    extents = [x - y for x, y in zip(bbox_max, bbox_min)]
    box = trimesh.creation.box(extents)
    box.apply_translation([(x + y) / 2 for x, y in zip(bbox_max, bbox_min)])

    facets_origin = box.facets_origin
    facets_normal = box.facets_normal
    clipped_mesh = mesh.slice_plane(facets_origin, -facets_normal, cap=cap)

    if export_path is not None:
        clipped_mesh.export(export_path)
        print(f"Mesh has been exported to {export_path}")

    return clipped_mesh


# ---------------------------------------------------------------------------
# Remote file streaming (SSH cat — no local copy)
# ---------------------------------------------------------------------------


def is_local_host(host: str) -> bool:
    """Return True if *host* resolves to the current machine."""
    local_hostname = socket.gethostname()
    # Strip user@ prefix if present
    remote = host.split("@", 1)[-1]
    try:
        remote_ips = {addr[4][0] for addr in socket.getaddrinfo(remote, None)}
        local_ips = {addr[4][0] for addr in socket.getaddrinfo(local_hostname, None)}
        local_ips.add("127.0.0.1")
        return bool(remote_ips & local_ips)
    except socket.gaierror:
        return False


def translate_path(
    path: Union[str, Path],
    path_map: List[Tuple[str, str]],
) -> Path:
    """Translate *path* by replacing the first matching prefix from *path_map*.

    *path_map* is a list of ``(src_prefix, dst_prefix)`` pairs.  Longer/more
    specific prefixes should come first so they match before shorter ones.
    Returns the path unchanged if no prefix matches.
    """
    s = str(path)
    for src, dst in path_map:
        if s.startswith(src):
            return Path(dst + s[len(src):])
    return Path(s)


def resolve_remote_file(
    path: Union[str, Path],
    host: str,
    path_map: Optional[List[Tuple[str, str]]] = None,
) -> Union[Path, io.BytesIO]:
    """Return *path* as a local ``Path`` if it exists, otherwise stream it from
    *host* via ``ssh cat`` into an in-memory ``BytesIO`` (no local copy saved).

    If *path_map* is given, the path is translated for the remote host using
    prefix substitution (see :func:`translate_path`).

    Raises ``FileNotFoundError`` when the file is missing both locally and on
    the remote, or when SSH streaming fails.
    """
    path = Path(path)
    if path.exists():
        return path

    if is_local_host(host):
        raise FileNotFoundError(
            f"Asset not found locally (same host as {host}): {path}"
        )

    remote_path = translate_path(path, path_map) if path_map else path

    remote_cmd = "cat " + shlex.quote(str(remote_path))
    result = subprocess.run(
        ["ssh", host, remote_cmd],
        capture_output=True,
    )
    if result.returncode != 0:
        stderr = result.stderr.decode(errors="replace").strip()
        raise FileNotFoundError(
            f"ssh cat failed for {host}:{remote_path}: {stderr}"
        )

    return io.BytesIO(result.stdout)


def resolve_remote_file_any(
    path: Union[str, Path],
    remote_hosts: List,
) -> Union[Path, io.BytesIO]:
    """Try to resolve *path* locally, then stream from each remote host.

    *remote_hosts* is a list of ``RemoteHost`` objects (from
    ``seen2scene.configs.opt``) with ``.ssh`` and ``.path_map`` attributes.
    Hosts that resolve to the local machine are skipped automatically.
    The first successful fetch wins.
    """
    path = Path(path)
    if path.exists():
        return path

    errors = []
    for rh in remote_hosts:
        if is_local_host(rh.ssh):
            continue
        try:
            return resolve_remote_file(path, rh.ssh, rh.path_map or None)
        except FileNotFoundError as e:
            errors.append(str(e))

    raise FileNotFoundError(
        f"File not found locally or on any remote host: {path}\n"
        + "\n".join(f"  - {e}" for e in errors)
    )


# ---------------------------------------------------------------------------
# Remote filesystem primitives (directory-level operations)
# ---------------------------------------------------------------------------


def reverse_translate_path(
    path: Union[str, Path],
    path_map: List[Tuple[str, str]],
) -> Path:
    """Inverse of :func:`translate_path`: host-native path → canonical path.

    *path_map* uses the same ``(canonical_prefix, host_prefix)`` convention as
    :class:`RemoteHost`.  This function matches on *host_prefix* (the second
    element) and rewrites to *canonical_prefix* (the first element).
    """
    s = str(path)
    for canonical, host in path_map:
        if s.startswith(host):
            return Path(canonical + s[len(host):])
    return Path(s)


_SSH_CONTROL_DIR = Path("/tmp") / f"ssh-mux-{os.getuid()}"


def _ssh_run(host: str, cmd: List[str], retries: int = 3) -> subprocess.CompletedProcess:
    """Run a command on *host* via SSH and return the CompletedProcess.

    Arguments are joined into a single shell command with :func:`shlex.quote`
    so that paths containing spaces or special characters (e.g.
    ``mesh_tsdf (Generation)_0.ply``) are handled correctly by the remote
    shell.

    Uses SSH ControlMaster multiplexing so the first call opens a persistent
    connection and subsequent calls reuse it (avoiding repeated auth/handshake).
    Retries up to *retries* times on timeout before giving up.
    """
    _SSH_CONTROL_DIR.mkdir(exist_ok=True)
    control_path = _SSH_CONTROL_DIR / f"%r@%h:%p"
    remote_cmd = " ".join(shlex.quote(str(a)) for a in cmd)
    for attempt in range(retries):
        try:
            return subprocess.run(
                [
                    "ssh",
                    "-o", "ConnectTimeout=10",
                    "-o", "BatchMode=yes",
                    "-o", "ServerAliveInterval=5",
                    "-o", f"ControlPath={control_path}",
                    "-o", "ControlMaster=auto",
                    "-o", "ControlPersist=300",
                    host,
                    remote_cmd,
                ],
                capture_output=True,
                timeout=30,
            )
        except subprocess.TimeoutExpired:
            if attempt < retries - 1:
                logger.warning(
                    f"SSH to {host} timed out (attempt {attempt + 1}/{retries}), retrying..."
                )
            else:
                logger.warning(f"SSH to {host} timed out after {retries} attempts: {remote_cmd}")
    return subprocess.CompletedProcess(
        args=["ssh", host, remote_cmd], returncode=1,
        stdout=b"", stderr=b"timeout",
        )


def remote_exists(
    path: Union[str, Path],
    remote_hosts: Optional[List] = None,
) -> bool:
    """Check if *path* exists locally or on any remote host."""
    path = Path(path)
    if path.exists():
        return True
    if not remote_hosts:
        return False
    for rh in remote_hosts:
        if is_local_host(rh.ssh):
            continue
        rp = translate_path(path, rh.path_map) if rh.path_map else path
        result = _ssh_run(rh.ssh, ["test", "-e", str(rp)])
        if result.returncode == 0:
            return True
    return False


def remote_is_dir(
    path: Union[str, Path],
    remote_hosts: Optional[List] = None,
) -> bool:
    """Check if *path* is a directory locally or on any remote host."""
    path = Path(path)
    if path.is_dir():
        return True
    if not remote_hosts:
        return False
    for rh in remote_hosts:
        if is_local_host(rh.ssh):
            continue
        rp = translate_path(path, rh.path_map) if rh.path_map else path
        result = _ssh_run(rh.ssh, ["test", "-d", str(rp)])
        if result.returncode == 0:
            return True
    return False


def remote_listdir(
    path: Union[str, Path],
    remote_hosts: Optional[List] = None,
    dirs_only: bool = False,
) -> List[str]:
    """List directory entries (names only) locally or from a remote host.

    Returns a sorted list of entry names.  When *dirs_only* is True, only
    subdirectory names are returned.
    """
    path = Path(path)
    if path.is_dir():
        if dirs_only:
            return sorted(d.name for d in path.iterdir() if d.is_dir())
        return sorted(e.name for e in path.iterdir())
    if not remote_hosts:
        return []
    for rh in remote_hosts:
        if is_local_host(rh.ssh):
            continue
        rp = translate_path(path, rh.path_map) if rh.path_map else path
        if dirs_only:
            cmd = ["find", str(rp), "-maxdepth", "1", "-mindepth", "1",
                   "-type", "d", "-printf", "%f\\n"]
        else:
            cmd = ["ls", "-1", str(rp)]
        result = _ssh_run(rh.ssh, cmd)
        if result.returncode == 0:
            names = result.stdout.decode().strip().split("\n")
            return sorted(n for n in names if n)
    return []


def remote_glob(
    path: Union[str, Path],
    pattern: str,
    remote_hosts: Optional[List] = None,
) -> List[Path]:
    """Glob for files matching *pattern* in *path*, locally or remotely.

    Returns canonical ``Path`` objects (reverse-translated from remote).
    """
    path = Path(path)
    if path.is_dir():
        return sorted(path.glob(pattern))
    if not remote_hosts:
        return []
    for rh in remote_hosts:
        if is_local_host(rh.ssh):
            continue
        rp = translate_path(path, rh.path_map) if rh.path_map else path
        cmd = ["find", str(rp), "-maxdepth", "1", "-name", pattern]
        result = _ssh_run(rh.ssh, cmd)
        if result.returncode == 0 and result.stdout.strip():
            lines = result.stdout.decode().strip().split("\n")
            out = []
            for line in lines:
                line = line.strip()
                if line:
                    canonical = reverse_translate_path(line, rh.path_map) if rh.path_map else Path(line)
                    out.append(canonical)
            if out:
                return sorted(out)
    return []


def remote_read_json(
    path: Union[str, Path],
    remote_hosts: Optional[List] = None,
) -> dict:
    """Read and parse a JSON file locally or by streaming from a remote host."""
    path = Path(path)
    if path.exists():
        with open(path) as f:
            return json.load(f)
    if not remote_hosts:
        raise FileNotFoundError(f"JSON file not found: {path}")
    data = resolve_remote_file_any(path, remote_hosts)
    if isinstance(data, io.BytesIO):
        return json.loads(data.read().decode())
    # resolve_remote_file_any returned a local Path
    with open(data) as f:
        return json.load(f)


def remote_load_mesh(
    path: Union[str, Path],
    remote_hosts: Optional[List] = None,
    **kwargs,
) -> trimesh.Trimesh:
    """Load a mesh locally or by streaming from a remote host.

    Extra *kwargs* are forwarded to ``trimesh.load``.
    """
    path = Path(path)
    if path.exists():
        return trimesh.load(str(path), **kwargs)
    if not remote_hosts:
        raise FileNotFoundError(f"Mesh file not found: {path}")
    data = resolve_remote_file_any(path, remote_hosts)
    if isinstance(data, io.BytesIO):
        # trimesh needs file_type hint when reading from a BytesIO
        file_type = kwargs.pop("file_type", path.suffix.lstrip("."))
        return trimesh.load(data, file_type=file_type, **kwargs)
    return trimesh.load(str(data), **kwargs)
