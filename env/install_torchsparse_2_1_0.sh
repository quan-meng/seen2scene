#!/usr/bin/env bash
set -euo pipefail

# Installs TorchSparse v2.1.0 from the upstream source branch.
# This package installs the Python module imported as `torchsparse`.
#
# Expected environment for this repo's sg conda env:
#   Linux, Python 3.11, PyTorch 2.4.x, CUDA/nvcc 12.0-12.4.

TORCHSPARSE_REPO="${TORCHSPARSE_REPO:-https://github.com/mit-han-lab/torchsparse.git}"
TORCHSPARSE_REF="${TORCHSPARSE_REF:-dev/v2.1_source_code}"
TORCHSPARSE_SRC="${TORCHSPARSE_SRC:-}"
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-7.0;7.5;8.0;8.6+PTX}"
MAX_JOBS="${MAX_JOBS:-$(nproc)}"
export TORCH_CUDA_ARCH_LIST MAX_JOBS

if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/nvcc" ]]; then
  CUDA_HOME="${CONDA_PREFIX}"
  PATH="${CUDA_HOME}/bin:${PATH}"
  export CUDA_HOME PATH
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
    raise SystemExit("ERROR: TorchSparse is expected to be built on Linux.")

py = sys.version_info
if (py.major, py.minor) != (3, 11):
    raise SystemExit(f"ERROR: sg expects Python 3.11, found {py.major}.{py.minor}.")

try:
    import torch
except Exception as exc:
    raise SystemExit(f"ERROR: cannot import torch in this environment: {exc}")

if not torch.__version__.startswith("2.4."):
    raise SystemExit(f"ERROR: sg expects PyTorch 2.4.x, found {torch.__version__}.")

if torch.version.cuda != "12.0":
    raise SystemExit(f"ERROR: sg expects PyTorch built with CUDA 12.0, found {torch.version.cuda}.")

nvcc = subprocess.check_output(["nvcc", "--version"], text=True)
match = re.search(r"release (\d+\.\d+)", nvcc)
if not match:
    raise SystemExit("ERROR: could not parse nvcc --version.")

cuda = tuple(map(int, match.group(1).split(".")))
if not ((12, 0) <= cuda <= (12, 4)):
    raise SystemExit(f"ERROR: sg expects nvcc CUDA 12.0-12.4, found {match.group(1)}.")

print(f"Environment OK: Python {py.major}.{py.minor}, torch {torch.__version__}, CUDA {match.group(1)}")
PY

python -m pip install --upgrade pip setuptools wheel ninja packaging

tmp_dir=""
if [[ -z "${TORCHSPARSE_SRC}" ]]; then
  tmp_dir="$(mktemp -d)"
  trap 'rm -rf "${tmp_dir}"' EXIT
  TORCHSPARSE_SRC="${tmp_dir}/torchsparse"
  git clone --depth 1 --branch "${TORCHSPARSE_REF}" "${TORCHSPARSE_REPO}" "${TORCHSPARSE_SRC}"
fi

python -m pip install --no-build-isolation --no-cache-dir --no-deps "${TORCHSPARSE_SRC}"

python - <<'PY'
import torchsparse

version = getattr(torchsparse, "__version__", "unknown")
if version != "2.1.0":
    raise SystemExit(f"ERROR: expected torchsparse 2.1.0, installed {version}.")

print("Installed torchsparse", version, "from", torchsparse.__file__)
PY
