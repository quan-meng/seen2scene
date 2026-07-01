import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = PROJECT_ROOT / "assets"


def _path_from_env(name: str, default: Path) -> Path:
    return Path(os.environ.get(name, default)).expanduser()


EXP_DIR = _path_from_env("SEEN2SCENE_EXP_DIR", Path("experiments"))
DATA_ROOT = _path_from_env("SEEN2SCENE_DATA_ROOT", PROJECT_ROOT / "data")

FRONT3D_RAW_DIR = _path_from_env("SEEN2SCENE_FRONT3D_RAW_DIR", DATA_ROOT / "3D-FRONT")
FRONT3D_DIR = _path_from_env("SEEN2SCENE_FRONT3D_DIR", DATA_ROOT / "3D-FRONT" / "v3")

SCANNETPP_RAW_DIR = _path_from_env(
    "SEEN2SCENE_SCANNETPP_RAW_DIR", DATA_ROOT / "scannetpp_raw"
)
SCANNETPP_LIDAR_DIR = _path_from_env(
    "SEEN2SCENE_SCANNETPP_LIDAR_DIR", DATA_ROOT / "scannetpp_lidar"
)
SCANNETPP_DIR = _path_from_env("SEEN2SCENE_SCANNETPP_DIR", DATA_ROOT / "scannetpp")

ARKITSCENES_RAW_DIR = _path_from_env(
    "SEEN2SCENE_ARKITSCENES_RAW_DIR", DATA_ROOT / "arkitscenes_raw"
)
ARKITSCENES_LIDAR_DIR = ARKITSCENES_RAW_DIR / "laser_scanner_point_clouds"
ARKITSCENES_TMP_DIR = _path_from_env(
    "SEEN2SCENE_ARKITSCENES_TMP_DIR", DATA_ROOT / "arkitscenes"
)
ARKITSCENES_DIR = ARKITSCENES_TMP_DIR / "fusion"

LLMSLAYOUT_DIR = _path_from_env("SEEN2SCENE_LLMSLAYOUT_DIR", DATA_ROOT / "LLMsLayout")
