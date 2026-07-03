#!/usr/bin/env bash
set -euo pipefail

# Installs the Seen2Scene VDBFusion fork against conda-provided C++ dependencies.

VDBFUSION_REPO="${VDBFUSION_REPO:-https://github.com/quan-meng/vdbfusion.git}"
VDBFUSION_REF="${VDBFUSION_REF:-seen2scene-v0.1.6}"
VDBFUSION_SRC="${VDBFUSION_SRC:-}"
MAX_JOBS="${MAX_JOBS:-$(nproc)}"
export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-${MAX_JOBS}}"

if ! command -v cmake >/dev/null 2>&1; then
  echo "ERROR: cmake is not on PATH. Install/activate the conda environment first." >&2
  exit 1
fi

python - <<'PY'
import sys

if sys.platform != "linux":
    raise SystemExit("ERROR: VDBFusion is expected to be built on Linux.")

py = sys.version_info
if (py.major, py.minor) != (3, 11):
    raise SystemExit(f"ERROR: sg expects Python 3.11, found {py.major}.{py.minor}.")

print(f"Environment OK: Python {py.major}.{py.minor}")
PY

python -m pip install --upgrade pip setuptools wheel ninja packaging

tmp_dir=""
if [[ -z "${VDBFUSION_SRC}" ]]; then
  tmp_dir="$(mktemp -d)"
  trap 'rm -rf "${tmp_dir}"' EXIT
  VDBFUSION_SRC="${tmp_dir}/vdbfusion"
  git clone --depth 1 --branch "${VDBFUSION_REF}" "${VDBFUSION_REPO}" "${VDBFUSION_SRC}"
fi

export CMAKE_PREFIX_PATH="${CONDA_PREFIX:-}:${CMAKE_PREFIX_PATH:-}"
export CMAKE_ARGS="${CMAKE_ARGS:-} -DUSE_SYSTEM_EIGEN3=ON -DUSE_SYSTEM_OPENVDB=ON -DUSE_SYSTEM_PYBIND11=ON -DCMAKE_POLICY_VERSION_MINIMUM=3.5"
python -m pip install --no-build-isolation --no-cache-dir "${VDBFUSION_SRC}"

python - <<'PY'
import vdbfusion

print("Installed vdbfusion from", vdbfusion.__file__)
PY
