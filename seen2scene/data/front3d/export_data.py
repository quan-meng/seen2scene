import blenderproc as bproc
import bpy
import os
import sys
import json
import math
import argparse
import numpy as np
import shutil

from common import load_front3d, export_tri_scene_mesh
from seen2scene.tools.camera_utils import CameraSampler
from seen2scene.tools.mesh_utils import visualize_cameras

parser = argparse.ArgumentParser(description="Render scene using BlenderProc")
parser.add_argument("--json_path", type=str, help="Path to the 3D front file")
parser.add_argument(
    "--future_path", type=str, help="Path to the 3D Future Model folder"
)
parser.add_argument(
    "--texture_path", type=str, help="Path to the 3D FRONT texture folder"
)
parser.add_argument("--out_dir", type=str, help="Path to the output directory")
parser.add_argument(
    "--infnite_floor", type=bool, help="Whether to use infinite floor", default=False
)
parser.add_argument("--sample_camera", type=bool, help="", default=False)
parser.add_argument("--image_height", type=int, help="Height of the image", default=512)
parser.add_argument("--image_width", type=int, help="Width of the image", default=512)
parser.add_argument(
    "--camera_fov",
    type=float,
    help="Field of view for the camera in degrees",
    default=60.0,
)
parser.add_argument(
    "--camera_area_ratio",
    type=float,
    help="Number of cameras to sample per unit",
    default=1.0,
)

args = parser.parse_args()

if not os.path.exists(args.future_path) or not os.path.exists(args.texture_path):
    raise Exception("One of the two folders does not exist!")

bproc.init()
mapping_file = bproc.utility.resolve_resource(
    os.path.join("front_3D", "3D_front_mapping.csv")
)

mapping = bproc.utility.LabelIdMapping.from_csv(mapping_file)

struct_objs, furni_objs = load_front3d(
    json_path=args.json_path,
    future_model_path=args.future_path,
    front_3D_texture_path=args.texture_path,
    label_mapping=mapping,
    # black_filter_path=os.path.join(os.path.dirname(__file__), "black_list.json"),
)
print(f"Loaded {len(struct_objs + furni_objs)} objects")

scene_name = os.path.basename(args.json_path).split(".")[0]
struct_objs, struc_is_triangles, _, struc_scene_box = export_tri_scene_mesh(
    struct_objs,
    scene_name=scene_name,
    filter_flags=["empty", "nan", "bound"],
    black_filter_path=os.path.join(os.path.dirname(__file__), "new_black_list.txt"),
)
furni_objs, furni_is_triangles, metadata, furni_scene_box = export_tri_scene_mesh(
    furni_objs,
    scene_name=scene_name,
    filter_flags=["empty", "nan", "size"],
    black_filter_path=os.path.join(os.path.dirname(__file__), "new_black_list.txt"),
    out_dir=args.out_dir,  # Pass out_dir
)

output_dir = os.path.join(args.out_dir, scene_name)
scene_box_min = np.minimum(struc_scene_box[0], furni_scene_box[0])
scene_box_max = np.maximum(struc_scene_box[1], furni_scene_box[1])
scene_box = np.stack([scene_box_min, scene_box_max])  # [2, 3]
metadata["scene_box"] = scene_box.tolist()

# Filter scene without structure
if len(struct_objs) == 0:
    print(f"Filter scene without structure: {scene_name}")
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir, ignore_errors=True)
        print(f"Deleted {output_dir}")
    exit()

# Filter empty scene
if len(furni_objs) == 0:
    print(f"Filter empty scene: {scene_name}")
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir, ignore_errors=True)
        print(f"Deleted {output_dir}")
    exit()

loaded_objects = struct_objs + furni_objs
is_triangles = struc_is_triangles + furni_is_triangles
loaded_object_names = [obj.get_name() for obj in loaded_objects]
# Delete objects that aren't in loaded_objects,
for obj in bpy.context.scene.objects:
    if obj.type == "MESH":
        if obj.name not in loaded_object_names:
            bpy.data.objects.remove(obj, do_unlink=True)
        else:
            index = loaded_object_names.index(obj.name)
            if not is_triangles[index]:
                print(f"Triangulating {obj.name} in scene {scene_name}")
                # Select the object
                bpy.context.view_layer.objects.active = obj
                bpy.ops.object.mode_set(mode="EDIT")  # Switch to Edit mode
                bpy.ops.mesh.select_all(action="SELECT")  # Select all mesh elements
                # Triangulate the mesh
                bpy.ops.mesh.quads_convert_to_tris()  # Convert quads to tris
                bpy.ops.object.mode_set(mode="OBJECT")  # Switch back to Object mode

room_sampler = bproc.sampler.Front3DPointInRoomSampler(
    front3d_objects=loaded_objects,
    amount_of_objects_needed_per_room=-1,
)
for floor_obj in room_sampler.used_floors:
    floor_bbox = floor_obj.get_bound_box()  # [8, 3]
    bbox_mins = np.min(floor_bbox, axis=0)  # [3]
    bbox_maxs = np.max(floor_bbox, axis=0)  # [3]
    floor_bbox = np.stack([bbox_mins, bbox_maxs])  # [2, 3]
    metadata["object_bboxes"].append(floor_bbox.tolist())
    metadata["object_categories"].append(floor_obj.get_cp("category_id"))
    metadata["object_names"].append(floor_obj.get_name())

os.makedirs(output_dir, exist_ok=True)
with open(os.path.join(output_dir, f"meta.json"), "w") as f:
    json.dump(metadata, f)
bpy.ops.wm.ply_export(filepath=os.path.join(output_dir, f"scene.ply"))

# semantic_out_dir = os.path.join(args.out_dir, "scene_semantic")
# os.makedirs(semantic_out_dir, exist_ok=True)
# semantic_ids = all_vertex_attributes["category_id"]  # [N]
# semantic_ids = semantic_ids.astype(np.uint32)
# np.save(os.path.join(semantic_out_dir, f"{scene_name}.npy"), semantic_ids)

# if args.sample_camera:
#     bproc.camera.set_intrinsics_from_blender_params(
#         lens=args.camera_fov * math.pi / 180.0,
#         lens_unit="FOV",
#         image_height=args.image_height,
#         image_width=args.image_width,
#     )

#     # Init bvh tree containing all mesh objects
#     bvh_tree = bproc.object.create_bvh_tree_multi_objects(
#         [o for o in loaded_objects if isinstance(o, bproc.types.MeshObject)]
#     )

#     special_objects = [obj.get_cp("category_id") for obj in loaded_objects]

#     camera_sampler = CameraSampler(
#         front3d_objects=loaded_objects,
#         amount_of_objects_needed_per_room=-1,
#         bvh_tree=bvh_tree,
#         special_objects=special_objects,
#         scene_name=scene_name,
#     )

#     cam_Ts, rooms_dict = camera_sampler.sample_random_poses(args.camera_area_ratio)

#     if rooms_dict and len(cam_Ts) > 0:
#         output_dir = os.path.join(
#             output_dir,
#             f"scans_fov_{args.camera_fov}_d_{args.camera_area_ratio:.1f}_r_{args.image_height}",
#         )
#         os.makedirs(output_dir, exist_ok=True)

#         with open(os.path.join(output_dir, "rooms.json"), "w") as f:
#             json.dump(rooms_dict, f, indent=4)

#         # Camera intrinsics and extrinsics
#         cameras = {"intrinsics": bproc.camera.get_intrinsics_as_K_matrix().tolist()}
#         cameras["extrinsics"] = [cam_T.tolist() for cam_T in cam_Ts]
#         with open(os.path.join(output_dir, "cameras.json"), "w") as f:
#             json.dump(cameras, f, indent=4)

#         visualize_cameras(
#             cam_Ts,
#             camera_scale=1.5,
#             export_path=os.path.join(output_dir, f"cameras.ply"),
#         )
