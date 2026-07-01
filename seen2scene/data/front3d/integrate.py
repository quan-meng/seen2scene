import json
import os
import glob
import numpy as np
from typing import Optional, List, Tuple

from seen2scene.tools.log_utils import get_logger
import seen2scene.tools.reconstruction.integration as integration

logger = get_logger(file_name=__file__)


def volumetric_fusion(
    scan_dir: str,
    out_dir: str,
    image_paths: Optional[List[str]] = None,
    semantic_paths: Optional[List[str]] = None,
    percents: List[float] = [1.0],
    voxel_size: float = 0.088,
    truncation: Optional[float] = None,
    space_carving: bool = True,
    with_mesh: bool = True,
    img_hw: Tuple[int, int] = (512, 512),
    min_depth: float = 0.0,
    max_depth: float = 100.0,
    each_room_at_least: int = 1,
):
    # sample frames with percent
    with open(os.path.join(scan_dir, "rooms.json")) as f:
        room_dict = json.load(f)
    frames = [[] for _ in range(len(percents))]
    export_pts = {}
    for room_idx, (room_str, room_data) in enumerate(room_dict.items()):
        frame_idx = np.array(room_data.pop("frame_idx"))
        prev_split = 0
        for i, percent in enumerate(percents):
            split = max(each_room_at_least, round(len(frame_idx) * percent))
            chunk = frame_idx[prev_split:split]
            frames[i].extend(chunk.tolist())
            prev_split = split

            if room_idx == len(room_dict) - 1:
                key = str(sum([len(frames[j]) for j in range(i + 1)]) - 1)
                export_pts[key] = percent

    frames = [frame for sublist in frames for frame in sublist]

    depth_paths = sorted(
        glob.glob(os.path.join(scan_dir, "depth", "*.png")),
        key=lambda x: int(os.path.basename(x).split(".")[0]),
    )

    depth_paths = [depth_paths[i] for i in frames]
    with open(os.path.join(scan_dir, "cameras.json")) as f:
        cameras_dict = json.load(f)
        extrinsics = np.array(cameras_dict["extrinsics"])[frames]
        intrinsics = np.array(cameras_dict["intrinsics"])
        intrinsics = intrinsics[None].repeat(len(extrinsics), axis=0)
    image_paths = [image_paths[i] for i in frames] if image_paths is not None else None
    semantic_paths = (
        [semantic_paths[i] for i in frames] if semantic_paths is not None else None
    )

    integration.vdb_integration(
        out_dir=out_dir,
        depth_paths=depth_paths,
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        image_paths=image_paths,
        semantic_paths=semantic_paths,
        export_pts=export_pts,
        voxel_size=voxel_size,
        truncation=truncation,
        space_carving=space_carving,
        with_mesh=with_mesh,
        img_hw=img_hw,
        min_depth=min_depth,
        max_depth=max_depth,
    )


