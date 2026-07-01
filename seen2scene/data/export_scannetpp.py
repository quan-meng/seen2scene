import sys
import os
import dataclasses
import tyro
import trimesh
import json
from typing import List
import glob
import numpy as np
import vdbfusion
from typing import *
import pandas as pd

from seen2scene.tools.mesh_utils import visualize_cameras
from seen2scene.tools.slurm_utils import submit_jobs, Slurm
from seen2scene.configs.dataset import ScannetPP
from seen2scene.tools.log_utils import get_logger

logger = get_logger(file_name=__file__)


def load_scene_mapping(scene_list_path: str) -> dict:
    """
    Load the scene mapping CSV file and create a dictionary mapping scene IDs to timestamps.

    Args:
        scene_list_path: Path to the scene mapping CSV file

    Returns:
        mapping_dict: Dictionary mapping scene IDs to timestamps
    """
    # Read the CSV file
    df = pd.read_csv(scene_list_path)

    # Create mapping dictionary using the correct column names
    # The first column is "Scene ID/Timestamp (Faro)"
    mapping_dict = {}
    for _, row in df.iterrows():
        timestamp = row["Scene ID/Timestamp (Faro)"]
        scene_id = row["release_id"]
        mapping_dict[scene_id] = timestamp

    return mapping_dict


def load_poses(pose_path: str) -> np.ndarray:
    with open(pose_path, "r") as f:
        lines = f.readlines()
        pose = [x for x in lines if x.strip()]
        pose = [line.strip().split(",") for line in pose]
        pose = np.array(pose).astype(np.float64)
        pose = pose.reshape(4, 4)
    return pose


def volumetric_fusion(
    lidar_dir: str,
    voxel_size: float,
    out_dir: str,
    percents: List[float] = [1.0],
    laser_fov: float = 300.0,
):
    trans_path = os.path.join(lidar_dir, "scans", "1mm", "transform.txt")
    trans = load_poses(trans_path)  # [4, 4]

    pose_paths = sorted(glob.glob(os.path.join(lidar_dir, "scans", "poses", "*.txt")))
    poses = []
    for pose_path in pose_paths:
        pose = load_poses(pose_path)  # [4, 4]
        pose = trans @ pose  # [4, 4]
        poses.append(pose)
    poses = np.stack(poses)  # [N, 4, 4]
    visualize_cameras(
        poses, camera_scale=2.0, export_path=os.path.join(out_dir, f"lasers.ply")
    )

    lidar_ids = [os.path.basename(pose_path).split(".")[0] for pose_path in pose_paths]
    lidar_paths = [
        os.path.join(lidar_dir, "scans", "separate", f"{lidar_id}.ply")
        for lidar_id in lidar_ids
    ]

    truncation = voxel_size * 3
    vdb_volume = vdbfusion.VDBVolume(voxel_size, truncation, space_carving=True)

    export_pts: dict[int, list] = {}
    for p in percents:
        export_pts.setdefault(max(0, int(p * len(lidar_paths)) - 1), []).append(p)

    for i, lidar_path in enumerate(lidar_paths):
        lidar_points = trimesh.load(lidar_path).vertices  # [N, 3]
        lidar_points = lidar_points.astype(np.float64)
        lidar_points_aligned = lidar_points @ trans[:3, :3].T + trans[:3, 3]
        vdb_volume.integrate(lidar_points_aligned, poses[i])

        if i in export_pts:
            for percent in export_pts[i]:
                vdb_path = os.path.join(
                    out_dir, f"fusion_p_{percent}_v_{voxel_size:.3f}.vdb"
                )
                ply_path = os.path.join(
                    out_dir, f"fusion_p_{percent}_v_{voxel_size:.3f}.ply"
                )
                vdb_volume.extract_vdb_grids(vdb_path)
                vert, tri = vdb_volume.extract_triangle_mesh(fill_holes=False)
                trimesh.Trimesh(vertices=vert, faces=tri, process=False).export(
                    ply_path
                )


def export_data(slurm: Slurm):
    def export_data_func(scene_dir: str, out_dir: str):
        seg_path = os.path.join(scene_dir, "scans", "segments_anno.json")
        if not os.path.exists(seg_path):
            print(f"Skip scene {scene_dir} because no segments_anno.json found!")
            return

        # Export scene bounding box
        laser_mesh_path = os.path.join(scene_dir, "scans", "mesh_aligned_0.05.ply")
        laser_mesh = trimesh.load(laser_mesh_path, process=False)

        scene_bounds = np.array(laser_mesh.bounds)  # [2, 3]
        scene_bounds = scene_bounds.tolist()

        # Export object bounding boxes
        with open(seg_path, "r") as f:
            segments_anno = json.load(f)

        object_bboxes = []
        object_names = []
        for object_meta in segments_anno["segGroups"]:
            # Access the nested 'obb' structure for min/max coordinates
            mins = object_meta["obb"]["min"]
            maxs = object_meta["obb"]["max"]
            object_bboxes.append([mins, maxs])
            object_names.append(object_meta["label"])

        # Add metadata
        scene_meta = {
            "scene_box": scene_bounds,
            "object_bboxes": object_bboxes,
            "object_names": object_names,
        }
        with open(os.path.join(out_dir, "meta.json"), "w") as f:
            json.dump(scene_meta, f)

        print(f"Export scene {scene_dir} done!")

    scene_dirs = sorted(glob.glob(os.path.join(ScannetPP.raw_dir, "data", "*")))
    fn_kwargs_list = []
    for scene_dir in scene_dirs:
        scene_name = os.path.basename(scene_dir)
        out_dir = os.path.join(ScannetPP.root_dir, scene_name)
        fn_kwargs_list.append({"scene_dir": scene_dir, "out_dir": out_dir})

    submit_jobs(
        fn_kwargs_list=fn_kwargs_list,
        fn=export_data_func,
        slurm_kwargs=dataclasses.asdict(slurm),
    )


def export_fusion(
    slurm: Slurm, voxel_size: float = 0.011, percents: Tuple[float] = (0.1, 0.5, 1.0)
):
    os.makedirs(ScannetPP.root_dir, exist_ok=True)

    fn_kwargs_share = {"voxel_size": voxel_size, "percents": percents}

    scene_list_dict = load_scene_mapping(ScannetPP.scene_list_path)

    scene_dirs = sorted(glob.glob(os.path.join(ScannetPP.raw_dir, "data", "*")))

    fn_kwargs_list = []
    for scene_dir in scene_dirs:
        scene_id = os.path.basename(scene_dir)
        scene_timestamp = scene_list_dict[scene_id]
        out_dir = os.path.join(ScannetPP.root_dir, scene_id)
        if all(
            os.path.exists(
                os.path.join(out_dir, f"fusion_p_{p}_v_{voxel_size:.3f}.vdb")
            )
            for p in list(percents)
        ):
            print(f"Skip scene {scene_dir} because fusion already exists!")
            continue

        os.makedirs(out_dir, exist_ok=True)
        lidar_dir = os.path.join(ScannetPP.lidar_dir, "data", scene_timestamp)
        assert os.path.exists(lidar_dir), f"Laser directory does not exist: {lidar_dir}"
        fn_kwargs_list.append({"lidar_dir": lidar_dir, "out_dir": out_dir})

    logger.info(f"Total {len(fn_kwargs_list)} tasks for volumetric fusion!")

    submit_jobs(
        fn_kwargs_share=fn_kwargs_share,
        fn_kwargs_list=fn_kwargs_list,
        fn=volumetric_fusion,
        slurm_kwargs=dataclasses.asdict(slurm),
    )


if __name__ == "__main__":
    logger.info("Starting ScannetPP data export process")
    tyro.extras.subcommand_cli_from_dict(
        {
            "export_data": export_data,
            "export_fusion": export_fusion,
        },
    )
    logger.info("ScannetPP data export process completed")
