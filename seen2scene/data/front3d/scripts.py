import subprocess
import os
import json
import logging

from seen2scene.tools.pyrender_utils import render_mesh, PerspectiveCamera
import numpy as np
from tqdm import tqdm
import glob
import cv2

logger = logging.getLogger(__name__)

blender_lib_path = os.environ.get("BLENDER_LIB_PATH", "")


def render_scene_w_pyrender(
    mesh_path: str,
    output_dir: str,
    camera_json: str,
    image_height: int = 512,
    image_width: int = 512,
    camera_fov: float = 60.0,
    min_depth: float = 0.0,
    max_depth: float = 100.0,
):
    try:
        with open(camera_json, "r") as f:
            camera_json = json.load(f)
        extrinsics = camera_json["extrinsics"]  # list of camera extrinsics

        depth_dir = os.path.join(output_dir, "depth")
        os.makedirs(depth_dir, exist_ok=True)

        existing_views = len(glob.glob(os.path.join(depth_dir, "*")))

        if existing_views == len(extrinsics):
            logger.info("Depth images already rendered, skipping...")
            return
        else:
            start_idx = max(0, existing_views - 1)  # avoid corrupted depth images
            logger.info(f"Rendering {start_idx} to {len(extrinsics)}")

        scene_name = os.path.basename(os.path.dirname(mesh_path))

        camera = PerspectiveCamera(yfov=camera_fov * np.pi / 180.0)

        for i in tqdm(
            range(start_idx, len(extrinsics)), desc=f"Rendering {scene_name}"
        ):
            depth = render_mesh(
                mesh_path,
                camera,
                np.array(extrinsics[i]),
                resolution=(image_height, image_width),
                only_depth=True,
            )
            depth[depth > 1e4] = 0.0  # 65504.0 or 1e10
            depth = (
                (depth - min_depth) / (max_depth - min_depth) * np.iinfo(np.uint16).max
            )
            depth = depth.astype(np.uint16)
            cv2.imwrite(f"{depth_dir}/{i:08d}.png", depth)

            # depth = depth.astype(np.uint32)
            # imageio.imwrite(f"{depth_dir}/{i:04d}.tiff", depth, compression="deflate")

    except Exception as e:
        logger.error(f"Error occurred: {str(e)}")


# run render_scene_blenderproc.py with subprocess
def render_scene_w_blender(**kwargs):
    # Remove the use of the current environment
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    # Configure environment variables to use Blender's internal libraries instead of Conda-installed ones
    if blender_lib_path:
        env["LD_LIBRARY_PATH"] = f"{blender_lib_path}:{env.get('LD_LIBRARY_PATH', '')}"

    cmd = [
        "blenderproc",
        "run",
        os.path.join(os.path.dirname(__file__), "render_w_blender.py"),
    ]
    for key, value in kwargs.items():
        cmd += [f"--{key}", str(value)]

    try:
        subprocess.run(
            cmd,
            check=True,
            env=env,
        )
    except Exception as e:
        logger.error(f"Error occurred: {str(e)}")
        logger.error(f"Command that failed: {' '.join(cmd)}")
        if isinstance(e, subprocess.CalledProcessError):
            logger.error(f"Return code: {e.returncode}")
            logger.error(f"Output: {e.output}")
            logger.error(f"Stderr: {e.stderr}")


def export_scene_data(**kwargs):
    # Remove the use of the current environment
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    # Configure environment variables to use Blender's internal libraries instead of Conda-installed ones
    if blender_lib_path:
        env["LD_LIBRARY_PATH"] = f"{blender_lib_path}:{env.get('LD_LIBRARY_PATH', '')}"

    cmd = [
        "blenderproc",
        "run",
        os.path.join(os.path.dirname(__file__), "export_data.py"),
    ]
    for key, value in kwargs.items():
        cmd += [f"--{key}", str(value)]

    try:
        subprocess.run(
            cmd,
            check=True,
            env=env,
        )
    except Exception as e:
        logger.error(f"Error occurred: {str(e)}")
        logger.error(f"Command that failed: {' '.join(cmd)}")
        if isinstance(e, subprocess.CalledProcessError):
            logger.error(f"Return code: {e.returncode}")
            logger.error(f"Output: {e.output}")
            logger.error(f"Stderr: {e.stderr}")
