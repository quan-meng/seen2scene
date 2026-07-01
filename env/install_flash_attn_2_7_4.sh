#!/usr/bin/env bash
set -euo pipefail

# Installs FlashAttention after the conda environment is created.
# flash-attn needs to see the already-installed torch package at build time, so
# it must be installed with --no-build-isolation rather than inside env/sg.yml.

FLASH_ATTN_VERSION="${FLASH_ATTN_VERSION:-2.7.4.post1}"
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
    raise SystemExit("ERROR: FlashAttention is expected to be installed on Linux.")

try:
    import torch
except Exception as exc:
    raise SystemExit(f"ERROR: cannot import torch in this environment: {exc}")

if not torch.__version__.startswith("2.4."):
    raise SystemExit(f"ERROR: expected PyTorch 2.4.x, found {torch.__version__}.")

if torch.version.cuda != "12.0":
    raise SystemExit(f"ERROR: expected PyTorch built with CUDA 12.0, found {torch.version.cuda}.")

nvcc = subprocess.check_output(["nvcc", "--version"], text=True)
match = re.search(r"release (\d+\.\d+)", nvcc)
if not match:
    raise SystemExit("ERROR: could not parse nvcc --version.")

cuda = tuple(map(int, match.group(1).split(".")))
if not ((12, 0) <= cuda <= (12, 4)):
    raise SystemExit(f"ERROR: expected nvcc CUDA 12.0-12.4, found {match.group(1)}.")

print(f"Environment OK: Python {sys.version_info.major}.{sys.version_info.minor}, torch {torch.__version__}, CUDA {match.group(1)}")
PY

python -m pip install --upgrade pip setuptools wheel ninja packaging
python -m pip install --no-build-isolation --no-cache-dir "flash-attn==${FLASH_ATTN_VERSION}"

python - <<'PY'
import flash_attn
print("Installed flash-attn", getattr(flash_attn, "__version__", "unknown"), "from", flash_attn.__file__)
PY
