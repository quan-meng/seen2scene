"""Lightweight remote host definitions - no heavy dependencies (torch, etc.).

Extracted from opt.py so that render scripts can import REMOTE_HOSTS without
pulling in the full training config stack (which can segfault when mixed with
tyro CLI parsing due to C-extension double-init issues).
"""
import dataclasses
from typing import List, Tuple


@dataclasses.dataclass
class RemoteHost:
    """A remote host reachable via SSH, with optional path prefix translation.

    When a remote host uses different mount points, *path_map* translates
    canonical prefixes to the host's native paths. More specific (longer)
    prefixes must come first.
    """

    ssh: str
    """SSH address, e.g. ``user@host``."""

    path_map: List[Tuple[str, str]] = dataclasses.field(default_factory=list)
    """``(canonical_prefix, host_prefix)`` pairs for path translation.
    Empty means canonical paths are valid on this host as-is."""


REMOTE_HOSTS: List[RemoteHost] = []
