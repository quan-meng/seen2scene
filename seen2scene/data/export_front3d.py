import os
import sys
import glob
import random
import json
import tyro
import re
import dataclasses
from typing import List

from seen2scene.tools.slurm_utils import submit_jobs, Slurm

from seen2scene.configs.dataset import Front3D
from seen2scene.tools.log_utils import get_logger
import front3d.scripts as front3d_scripts
from front3d.integrate import volumetric_fusion


logger = get_logger(file_name=__file__)


def parse_scan_dir(scan_dir: str) -> List[str]:
    match = re.search(r"fov_(\d+\.\d+)_d_\d+\.\d+_r_(\d+)", scan_dir)
    if match:
        camera_fov = float(match.group(1))
        resolution = int(match.group(2))
    else:
        raise ValueError(f"Invalid camera_dir format: {scan_dir}")

    return camera_fov, resolution


def export_data(
    slurm: Slurm,
    camera_fov: float = 60.0,
    resolution: int = 512,
    camera_area_ratio: float = 8.0,
    infnite_floor: bool = False,
    sample_camera: bool = False,
    num_workers: int = 8,
) -> None:
    json_paths = sorted(glob.glob(os.path.join(Front3D.json_dir, "*.json")))

    fn_kwargs_share = {
        "infnite_floor": infnite_floor,
        "future_path": Front3D.future_path,
        "texture_path": Front3D.texture_path,
        "camera_area_ratio": camera_area_ratio,
        "camera_fov": camera_fov,
        "image_height": resolution,
        "sample_camera": sample_camera,
        "out_dir": Front3D.root_dir,
    }

    fn_kwargs_list = []
    for json_path in json_paths:
        fn_kwargs_list.append({"json_path": json_path})

    logger.info(f"Exporting mesh and cameras for {len(fn_kwargs_list)} scenes")
    submit_jobs(
        fn_kwargs_list=fn_kwargs_list,
        fn=front3d_scripts.export_scene_data,
        slurm_kwargs=dataclasses.asdict(slurm),
        fn_kwargs_share=fn_kwargs_share,
        num_workers=num_workers,
    )


def export_scans(
    slurm: Slurm,
    camera_dir: str = "scans_fov_60.0_d_8.0_r_512",
    depth_only: bool = True,
    renderer: str = "blender",
) -> None:
    camera_fov, resolution = parse_scan_dir(camera_dir)

    fn_kwargs_share = {
        "camera_fov": camera_fov,
        "image_height": resolution,
        "image_width": resolution,
    }

    fn_kwargs_list = []
    if renderer == "pyrender":
        from .front3d.scripts import render_scene_w_pyrender as scene_render

        fn_kwargs_share = {
            "min_depth": Front3D.min_depth,
            "max_depth": Front3D.max_depth,
        }

        scan_dirs = glob.glob(os.path.join(Front3D.root_dir, "*", camera_dir))
        random.shuffle(scan_dirs)

        for i, scan_dir in enumerate(scan_dirs):
            camera_json = os.path.join(scan_dir, "cameras.json")
            if os.path.exists(camera_json):
                fn_kwargs_list.append(
                    {
                        "output_dir": scan_dir,
                        "mesh_path": os.path.join(
                            os.path.dirname(scan_dir),
                            f"scene.ply",
                        ),
                        "camera_json": camera_json,
                    }
                )
    else:
        from .front3d.scripts import render_scene_w_blender as scene_render

        json_paths = sorted(glob.glob(os.path.join(Front3D.json_dir, "*.json")))

        fn_kwargs_share = {
            "future_path": Front3D.future_path,
            "texture_path": Front3D.texture_path,
            "cctextures_dir": Front3D.cctextures_dir,
            "render_rgb": not depth_only,
            "render_segmentation": not depth_only,
        }
        for i in range(len(fn_kwargs_list)):
            fn_kwargs_list[i]["json_path"] = json_paths[i]

    logger.info(f"Rendering {len(fn_kwargs_list)} scenes")

    submit_jobs(
        fn_kwargs_list=fn_kwargs_list,
        fn=scene_render,
        slurm_kwargs=dataclasses.asdict(slurm),
        fn_kwargs_share=fn_kwargs_share,
    )


def export_fusion(
    slurm: Slurm,
    voxel_size: float = 0.011,
    percents: List[float] = [0.1, 0.5, 1.0],
    scan_dir: str = "scans_fov_60.0_d_8.0_r_512",
) -> None:
    front3d = Front3D()

    _, resolution = parse_scan_dir(scan_dir)

    fn_kwargs_share = {
        "voxel_size": voxel_size,
        "truncation": voxel_size * 3,
        "percents": percents,
        "space_carving": True,
        "img_hw": [resolution, resolution],
        "min_depth": front3d.min_depth,
        "max_depth": front3d.max_depth,
    }

    if len(glob.glob(os.path.join(front3d.root_dir, "*", scan_dir, "semantic"))) > 0:
        semantic_flag = True
    else:
        semantic_flag = False

    scan_dirs = glob.glob(os.path.join(front3d.root_dir, "*", scan_dir))
    random.shuffle(scan_dirs)

    fusion_dir = front3d.fusion_dir[f"{voxel_size:.3f}"]
    os.makedirs(fusion_dir, exist_ok=True)

    fn_kwargs_list = []
    for scan_dir_i in scan_dirs:
        scene_name = scan_dir_i.split("/")[-2]
        out_dir_i = os.path.join(fusion_dir, scene_name)

        if not os.path.exists(os.path.join(scan_dir_i, "cameras.json")):
            continue

        depth_dir = os.path.join(scan_dir_i, "depth")
        with open(os.path.join(scan_dir_i, "cameras.json")) as camera_file:
            extrinsics = json.load(camera_file)["extrinsics"]

        if len(extrinsics) == 0 or len(extrinsics) != len(
            glob.glob(os.path.join(depth_dir, "*.png"))
        ):
            continue

        if os.path.exists(
            os.path.join(out_dir_i, f"fusion_p_{percents[-1]}_v_{voxel_size:.3f}.vdb")
        ):
            continue

        fn_kwargs = {"out_dir": out_dir_i, "scan_dir": scan_dir_i}

        if semantic_flag:
            fn_kwargs["semantic_paths"] = sorted(
                glob.glob(os.path.join(scan_dir_i, "semantic", "*.png"))
            )

        fn_kwargs_list.append(fn_kwargs)

    logger.info(f"Volumetric fusion for {len(fn_kwargs_list)} scenes")

    submit_jobs(
        fn_kwargs_list=fn_kwargs_list,
        fn=volumetric_fusion,
        slurm_kwargs=dataclasses.asdict(slurm),
        fn_kwargs_share=fn_kwargs_share,
    )


if __name__ == "__main__":
    logger.info("Starting Front3D data export process")
    tyro.extras.subcommand_cli_from_dict(
        {
            "export_data": export_data,
            "export_scans": export_scans,
            "export_fusion": export_fusion,
        },
    )
    logger.info("Front3D data export process completed")
