#!/usr/bin/env bash
set -euo pipefail

# Installs ASWF OpenVDB fVDB v0.2.1 from the tagged repository.
# This package installs the Python module imported as `fvdb`.
#
# Expected environment for fvdb_v0.2.1:
#   Linux, Python 3.10-3.12, PyTorch 2.4.x, CUDA/nvcc 12.0-12.4.

FVDB_REPO="${FVDB_REPO:-https://github.com/AcademySoftwareFoundation/openvdb.git}"
FVDB_TAG="${FVDB_TAG:-fvdb_v0.2.1}"
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-7.0;7.5;8.0;8.6+PTX}"
MAX_JOBS="${MAX_JOBS:-$(nproc)}"
export TORCH_CUDA_ARCH_LIST MAX_JOBS

if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/nvcc" ]]; then
  CUDA_HOME="${CONDA_PREFIX}"
  PATH="${CUDA_HOME}/bin:${PATH}"
  export CUDA_HOME PATH
fi

if [[ -n "${CONDA_PREFIX:-}" ]]; then
  LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
  LD_PRELOAD="${CONDA_PREFIX}/lib/libtbb.so${LD_PRELOAD:+:${LD_PRELOAD}}"
  export LD_LIBRARY_PATH LD_PRELOAD
  mkdir -p "${CONDA_PREFIX}/etc/conda/activate.d"
  {
    echo 'export SEEN2SCENE_OLD_LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"'
    echo 'export SEEN2SCENE_OLD_LD_PRELOAD="${LD_PRELOAD:-}"'
    echo 'export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"'
    echo 'export LD_PRELOAD="${CONDA_PREFIX}/lib/libtbb.so${LD_PRELOAD:+:${LD_PRELOAD}}"'
  } > "${CONDA_PREFIX}/etc/conda/activate.d/seen2scene-fvdb.sh"
fi

if ! command -v nvcc >/dev/null 2>&1; then
  echo "ERROR: nvcc is not on PATH. Activate/install a CUDA toolkit environment first." >&2
  exit 1
fi

if [[ -z "${CUDA_HOME:-}" ]]; then
  CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
  export CUDA_HOME
fi

python - <<'PY'
import re
import subprocess
import sys

if sys.platform != "linux":
    raise SystemExit("ERROR: fVDB v0.2.1 only supports Linux.")

py = sys.version_info
if not ((py.major, py.minor) >= (3, 10) and (py.major, py.minor) <= (3, 12)):
    raise SystemExit(f"ERROR: Python {py.major}.{py.minor} is not supported; use Python 3.10-3.12.")

try:
    import torch
except Exception as exc:
    raise SystemExit(f"ERROR: cannot import torch in this environment: {exc}")

if not torch.__version__.startswith("2.4."):
    raise SystemExit(f"ERROR: fVDB v0.2.1 expects PyTorch 2.4.x, found {torch.__version__}.")

nvcc = subprocess.check_output(["nvcc", "--version"], text=True)
match = re.search(r"release (\d+\.\d+)", nvcc)
if not match:
    raise SystemExit("ERROR: could not parse nvcc --version.")

cuda = tuple(map(int, match.group(1).split(".")))
if not ((12, 0) <= cuda <= (12, 4)):
    raise SystemExit(f"ERROR: fVDB v0.2.1 expects CUDA 12.0-12.4, found {match.group(1)}.")

print(f"Environment OK: Python {py.major}.{py.minor}, torch {torch.__version__}, CUDA {match.group(1)}")
PY

python -m pip install --upgrade pip setuptools wheel ninja packaging

if python - <<'PY'
import fvdb
print("Installed fvdb", getattr(fvdb, "__version__", "unknown"), "from", fvdb.__file__)
PY
then
  exit 0
fi

python -m pip install --no-build-isolation --no-cache-dir \
  "git+${FVDB_REPO}@${FVDB_TAG}#subdirectory=fvdb"

python - <<'PY'
import fvdb
print("Installed fvdb", getattr(fvdb, "__version__", "unknown"), "from", fvdb.__file__)
PY
