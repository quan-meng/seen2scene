import blenderproc as bproc
from typing import Optional
from collections import OrderedDict
import numpy as np


class CameraSampler(bproc.sampler.Front3DPointInRoomSampler):
    def __init__(
        self,
        front3d_objects,
        amount_of_objects_needed_per_room,
        bvh_tree,
        special_objects,
        scene_name: str,
        random_seed: Optional[int] = None,
    ):
        super().__init__(front3d_objects, amount_of_objects_needed_per_room)
        self.rng = np.random.RandomState(random_seed)
        self.scene_name = scene_name
        self.bvh_tree = bvh_tree
        self.special_objects = special_objects

    def calculate_views_per_room(self, camera_area_ratio, min_views_per_room=1):
        rooms_dict = OrderedDict()
        total_area = 0
        # Calculate areas and total area
        for idx, floor_obj in enumerate(self.used_floors):
            bounding_box = floor_obj.get_bound_box()[:, :2]  # [2, 3] -> [2, 2]
            min_corner = np.min(bounding_box, axis=0)
            max_corner = np.max(bounding_box, axis=0)
            area = (max_corner[0] - min_corner[0]) * (max_corner[1] - min_corner[1])

            if area * camera_area_ratio < min_views_per_room:
                continue

            rooms_dict |= {
                f"room_{idx}": {
                    "area": area,
                    "n_views": 0,
                    "bbox": bounding_box.tolist(),
                }
            }
            total_area += area

        for room_str, room_data in rooms_dict.items():
            n_views_per_room = max(
                min_views_per_room, int(round(room_data["area"] * camera_area_ratio))
            )
            rooms_dict[room_str]["n_views"] = n_views_per_room

        return rooms_dict

    def sample_rotation(self):
        return (
            np.array(
                [
                    self.rng.uniform(30, 150.0),
                    0.0,
                    self.rng.uniform(0, 360.0),
                ]
            )
            * np.pi
            / 180.0
        )

    def sample_loc_step(
        self,
        floor_idx: int,
        min_height: float = 1.5,
        max_height: float = 2.0,
    ) -> np.ndarray:
        """Samples a point inside one of the loaded Front3d rooms.

        The points are uniformly sampled along x/y over all rooms.
        :return: The sampled point.
        """
        floor_obj = self.used_floors[floor_idx]
        bounds = floor_obj.get_bound_box()
        min_corner = np.min(bounds, axis=0)
        max_corner = np.max(bounds, axis=0)

        # Use rng for uniform sampling instead of random
        point = np.concatenate(
            [
                self.rng.uniform(min_corner[0], max_corner[0], 1),
                self.rng.uniform(min_corner[1], max_corner[1], 1),
                floor_obj.get_location()[2]
                + self.rng.uniform(min_height, max_height, 1),
            ]
        )  # [3]

        if floor_obj.position_is_above_object(point, check_no_objects_in_between=False):
            return point

        return None

    def sample_camera_step(
        self,
        floor_idx,
        proximity_checks,
        max_tries_multiplier: int = 1000,
        special_objects_weight: float = 10.0,
        scene_coverage_score_min: float = 0.5,
    ):
        tries = 0
        while tries < max_tries_multiplier:
            tries += 1

            # Sample point inside house
            location = self.sample_loc_step(floor_idx=floor_idx)
            if location is None:
                continue

            # Sample rotation (fix around X and Y axis)
            rotation = self.sample_rotation()

            cam2world_matrix = bproc.math.build_transformation_mat(location, rotation)

            # Check if the pose is valid
            cov_score = bproc.camera.scene_coverage_score(
                cam2world_matrix,
                self.special_objects,
                special_objects_weight=special_objects_weight,
            )
            proximity_check = bproc.camera.perform_obstacle_in_view_check(
                cam2world_matrix, proximity_checks, self.bvh_tree
            )

            is_valid = (cov_score > scene_coverage_score_min) and proximity_check

            if is_valid:
                return cam2world_matrix

        return None

    def sample_random_poses(self, camera_area_ratio, max_tries_multiplier: int = 100):
        cam_Ts = []
        proximity_checks = {
            "min": 0.1,
            "no_background": True,
        }

        rooms_dict = self.calculate_views_per_room(camera_area_ratio)

        # new n_views
        n_views = sum(room_data["n_views"] for room_data in rooms_dict.values())
        tries = 0
        room_poses = [0] * len(self.used_floors)  # Track poses per room
        while sum(room_poses) < n_views and tries < (max_tries_multiplier * n_views):
            tries += 1
            for floor_str in rooms_dict.keys():
                floor_idx = int(floor_str.split("_")[-1])
                if room_poses[floor_idx] >= rooms_dict[floor_str]["n_views"]:
                    continue

                cam2world_matrix = self.sample_camera_step(
                    floor_idx,
                    proximity_checks,
                    special_objects_weight=1.0,
                    scene_coverage_score_min=1e-3,
                )
                if cam2world_matrix is None:
                    continue

                curr_view_idx = sum(room_poses)
                rooms_dict[f"room_{floor_idx}"].setdefault("frame_idx", []).append(
                    curr_view_idx
                )
                cam_Ts.append(cam2world_matrix)
                room_poses[floor_idx] += 1

        if sum(room_poses) < n_views:
            print(
                f"Failed to sample all views in scene: {self.scene_name}. Only {sum(room_poses)} views sampled."
            )

        return cam_Ts, rooms_dict




