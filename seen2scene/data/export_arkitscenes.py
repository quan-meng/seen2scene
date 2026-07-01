import sys
import os
import dataclasses
import csv
import tyro
from typing import Dict, List
import glob
import tarfile
import urllib.request
import json
from collections import defaultdict
import numpy as np
import trimesh
import vdbfusion

from seen2scene.configs.dataset import ARKitScenes
from seen2scene.tools.log_utils import get_logger
from seen2scene.tools.slurm_utils import submit_jobs, Slurm
from seen2scene.tools.mesh_utils import detect_floor_z, bboxes2mesh, cameras2mesh

logger = get_logger(file_name=__file__)


def load_visit_to_video_ids(metadata_csv: str) -> Dict[str, List[str]]:
    """
    Build a mapping from visit_id -> list of video_ids using 3dod/metadata.csv.
    Rows with missing/NA visit_id or video_id are skipped. Video ids are kept
    unique per visit_id while preserving the file order.
    """
    visit_to_videos: Dict[str, List[str]] = defaultdict(list)
    with open(metadata_csv, newline="") as f:
        reader = csv.reader(f)
        next(reader, None)  # skip header if present
        for row in reader:
            if len(row) < 2:
                continue
            video_id, visit_id = row[0].strip(), row[1].strip()
            if not video_id or not visit_id or visit_id == "NA":
                continue
            if video_id not in visit_to_videos[visit_id]:
                visit_to_videos[visit_id].append(video_id)
    return dict(visit_to_videos)


def download_instances_json(url: str, scan_name: str, tar_path: str, json_path: str):
    urllib.request.urlretrieve(url, tar_path)

    with tarfile.open(tar_path, "r") as tar:
        try:
            member = tar.getmember(f"{scan_name}/world.gt/instances.json")
        except KeyError as exc:
            raise FileNotFoundError(
                f"world.gt/instances.json not found in {tar_path}"
            ) from exc

        extracted = tar.extractfile(member)
        if extracted is None:
            raise RuntimeError(f"Failed to extract {member.name} from {tar_path}")

        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        with open(json_path, "wb") as f:
            f.write(extracted.read())

        # Delete the tar file
        os.remove(tar_path)


def download_1mca(slurm: Slurm):
    """
    Download CA-1M tar files listed in train/val manifests, extract only
    world.gt/instances.json, then delete the tar to save space.
    """
    out_dir = os.path.join(ARKitScenes.tmp_dir, "data")
    os.makedirs(out_dir, exist_ok=True)

    urls = []
    list_dir = os.path.abspath("assets/ca-1m")
    for split in ["train", "val"]:
        list_path = os.path.join(list_dir, f"{split}.txt")
        with open(list_path, "r") as f:
            urls.extend([line.strip() for line in f.readlines() if line.strip()])

    fn_kwargs_list = []
    for url in urls:
        # ca1m-train-42444692 to 42444692
        scan_name = os.path.basename(url).replace(".tar", "")
        scan_name = scan_name.split("-")[-1]
        tar_path = os.path.join(out_dir, f"{scan_name}.jar")
        json_path = os.path.join(out_dir, f"{scan_name}.json")

        if os.path.exists(json_path):
            logger.info(f"Skip {scan_name}: {json_path} already exists.")
            continue

        fn_kwargs_list.append(
            {
                "url": url,
                "scan_name": scan_name,
                "tar_path": tar_path,
                "json_path": json_path,
            }
        )

    logger.info(f"Total {len(fn_kwargs_list)} tasks for downloading instances.json!")

    submit_jobs(
        fn_kwargs_list=fn_kwargs_list,
        fn=download_instances_json,
        slurm_kwargs=dataclasses.asdict(slurm),
    )


def volumetric_fusion(
    lidar_dir: str,
    voxel_size: float,
    out_dir: str,
    percents: List[float] = [1.0],
    laser_fov: float = 300.0,
):
    os.makedirs(out_dir, exist_ok=True)

    lidar_paths = sorted(glob.glob(os.path.join(lidar_dir, "*.ply")))
    lidar_ids = [os.path.basename(x).split(".")[0] for x in lidar_paths]
    pose_paths = [os.path.join(lidar_dir, f"{id}_pose.txt") for id in lidar_ids]

    poses = []
    for pose_path in pose_paths:
        with open(pose_path, "r") as f:
            lines = f.readlines()
            pose = [x for x in lines if x.strip()]
            pose = [line.strip().split(",") for line in pose]
            pose = np.array(pose).astype(np.float64)
            pose = pose.T
        poses.append(pose)
    poses = np.stack(poses)  # [N, 4, 4]
    if poses.ndim == 2:
        poses = poses[None]  # [1, 4, 4]

    cameras2mesh(
        poses,
        camera_scale=2.0,
        export_path=os.path.join(out_dir, f"lasers.ply"),
        fov=laser_fov,
    )

    truncation = voxel_size * 3
    vdb_volume = vdbfusion.VDBVolume(voxel_size, truncation, space_carving=True)

    export_pts: dict[int, list] = {}
    for p in percents:
        export_pts.setdefault(max(0, int(p * len(lidar_paths)) - 1), []).append(p)

    for i, lidar_path in enumerate(lidar_paths):
        lidar_points = trimesh.load(lidar_path).vertices  # [N, 3]
        lidar_points = lidar_points.astype(np.float64)
        vdb_volume.integrate(lidar_points, poses[i])

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


def export_fusion(
    slurm: Slurm, voxel_size: float = 0.011, percents: List[float] = [0.1, 1.0]
):
    out_dir = os.path.join(ARKitScenes.tmp_dir, "fusion")
    os.makedirs(out_dir, exist_ok=True)

    fn_kwargs_share = {"voxel_size": voxel_size, "percents": percents}

    lidar_dirs = sorted(glob.glob(os.path.join(ARKitScenes.lidar_dir, "*")))
    import ipdb

    ipdb.set_trace()

    fn_kwargs_list = []
    for lidar_dir in lidar_dirs:
        scene_id = os.path.basename(lidar_dir)
        out_dir_i = os.path.join(out_dir, scene_id)
        if all(
            os.path.exists(
                os.path.join(out_dir_i, f"fusion_p_{p}_v_{voxel_size:.3f}.vdb")
            )
            for p in percents
        ):
            print(f"Skip scene {out_dir_i} because fusion already exists!")
            continue
        lidar_paths = sorted(glob.glob(os.path.join(lidar_dir, "*.ply")))
        lidar_ids = [os.path.basename(x).split(".")[0] for x in lidar_paths]
        pose_paths = [os.path.join(lidar_dir, f"{id}_pose.txt") for id in lidar_ids]
        pose_paths_wrong = sorted(glob.glob(os.path.join(lidar_dir, "*.txt")))
        if len(pose_paths_wrong) == 0 or len(lidar_paths) == 0:
            continue
        if pose_paths == pose_paths_wrong:
            continue

        fn_kwargs_list.append({"lidar_dir": lidar_dir, "out_dir": out_dir_i})
    logger.info(f"Total {len(fn_kwargs_list)} tasks for volumetric fusion!")

    submit_jobs(
        fn_kwargs_share=fn_kwargs_share,
        fn_kwargs_list=fn_kwargs_list,
        fn=volumetric_fusion,
        slurm_kwargs=dataclasses.asdict(slurm),
    )


def export_data(slurm: Slurm, voxel_size: float = 0.011):
    """
    Convert CA-1M instances.json files into meta.json in the same format as
    export_scannetpp.py (axis-aligned scene_box and per-object bboxes + names).
    Expects instances jsons already downloaded under ARKitScenes.tmp_dir/data.
    """

    fusion_dir = ARKitScenes().fusion_dir[f"{voxel_size:.3f}"]
    json_dir = os.path.join(ARKitScenes.tmp_dir, "data")
    assert os.path.exists(json_dir), f"Data dir not found: {json_dir}"

    def convert_instances(json_paths: List[str], out_dir: str, ply_path: str):
        scene_mesh = trimesh.load(ply_path, process=False)
        scene_bounds = np.array(scene_mesh.bounds)  # [2, 3]
        scene_bounds = scene_bounds.tolist()

        floor_z = detect_floor_z(scene_mesh, method="combined")
        scene_bounds[0][2] = floor_z

        instances = []
        for json_path in json_paths:
            with open(json_path, "r") as f:
                instances.append(json.load(f))
        assert all([instances[0] == x for x in instances]), "instances are not aligned!"

        instances = instances[0]

        object_bboxes, object_names = [], []
        for obj in instances:
            # corners is a list of 8 vertices; take axis-aligned min/max.
            corners = np.array(obj["corners"], dtype=np.float32)  # [8, 3]
            mins = corners.min(axis=0).tolist()
            maxs = corners.max(axis=0).tolist()

            object_bboxes.append([mins, maxs])
            object_names.append(obj.get("category", "void"))

        scene_meta = {
            "scene_box": scene_bounds,
            "object_bboxes": object_bboxes,
            "object_names": object_names,
        }

        mesh = bboxes2mesh(bboxes=np.array(object_bboxes, dtype=np.float32))
        mesh.export(os.path.join(out_dir, "bboxes_mesh.ply"))

        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "meta.json"), "w") as f:
            json.dump(scene_meta, f)

    csv_path = os.path.join(ARKitScenes.tmp_dir, "3dod/metadata.csv")
    mapping_dict = load_visit_to_video_ids(csv_path)

    fusion_dirs = sorted(glob.glob(os.path.join(fusion_dir, "*")))
    fn_kwargs_list = []
    for fusion_dir in fusion_dirs:
        visit_id = os.path.basename(fusion_dir)
        if visit_id not in mapping_dict:
            logger.info(f"Visit {visit_id} not found in 3dod/metadata.csv!")
            continue
        # if os.path.exists(os.path.join(fusion_dir, "meta.json")):
        #     logger.info(f"Skip {fusion_dir}: meta already exists.")
        #     continue

        ply_path = os.path.join(fusion_dir, f"fusion_p_1.0_v_{voxel_size:.3f}.ply")
        if not os.path.exists(ply_path):
            logger.info(f"PLY file {ply_path} not found!")
            continue
        video_list = mapping_dict[visit_id]
        json_paths = []
        for video_id in video_list:
            json_path = os.path.join(json_dir, f"{video_id}.json")
            if os.path.exists(json_path):
                json_paths.append(json_path)
            else:
                logger.info(
                    f"JSON file video: {video_id} of visit: {visit_id} not found!"
                )
                continue
        if len(json_paths) == 0:
            logger.info(f"No JSON files found for visit: {visit_id}!")
            continue

        fn_kwargs_list.append(
            {"json_paths": json_paths, "out_dir": fusion_dir, "ply_path": ply_path}
        )

    logger.info(f"Total {len(fn_kwargs_list)} tasks for converting meta!")

    submit_jobs(
        fn_kwargs_list=fn_kwargs_list,
        fn=convert_instances,
        slurm_kwargs=dataclasses.asdict(slurm),
    )


if __name__ == "__main__":
    logger.info("Starting ARKitScenes data export process")
    tyro.extras.subcommand_cli_from_dict(
        {
            "export_fusion": export_fusion,
            "download_1mca": download_1mca,
            "export_data": export_data,
        },
    )
    logger.info("ARKitScenes data export process completed")
