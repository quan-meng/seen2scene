import blenderproc as bproc
import os
import random
import numpy as np
import argparse
import imageio
import math
import json
import bpy
import sys

from common import load_front3d

parser = argparse.ArgumentParser(description="Render scene using BlenderProc")
parser.add_argument(
    "--future_path", type=str, help="Path to the 3D Future Model folder"
)
parser.add_argument(
    "--texture_path", type=str, help="Path to the 3D FRONT texture folder"
)
parser.add_argument("--json_path", type=str, help="Path to the 3D front file")
parser.add_argument("--output_dir", type=str, help="Path to the output directory")
parser.add_argument(
    "--cctextures_dir", type=str, help="Path to CCTextures folder", default=None
)
parser.add_argument(
    "--camera_fov",
    type=float,
    help="Field of view for the camera in degrees",
    default=60.0,
)
parser.add_argument(
    "--render_rgb",
    type=lambda x: x.lower() == "true",
    help="Whether to render RGB images",
    default=False,
)
parser.add_argument(
    "--render_segmentation",
    type=lambda x: x.lower() == "true",
    help="Whether to render segmentation maps",
    default=False,
)
parser.add_argument("--image_height", type=int, help="Height of the image", default=512)
parser.add_argument("--image_width", type=int, help="Width of the image", default=512)
args = parser.parse_args()

if not os.path.exists(args.future_path) or not os.path.exists(args.texture_path):
    raise Exception("One of the two folders does not exist!")

bproc.init()

mapping_file = bproc.utility.resolve_resource(
    os.path.join("front_3D", "3D_front_mapping.csv")
)

mapping = bproc.utility.LabelIdMapping.from_csv(mapping_file)

# load the front 3D objects
struct_objs, furni_objs = load_front3d(
    json_path=args.json_path,
    future_model_path=args.future_path,
    front_3D_texture_path=args.texture_path,
    label_mapping=mapping,
    black_filter_path=os.path.join(os.path.dirname(__file__), "black_list.json"),
)
loaded_objects = struct_objs + furni_objs

# Delete objects that aren't in loaded_objects, make rendering and mesh data consistent
loaded_object_names = {obj.get_name() for obj in loaded_objects}
for obj in bpy.context.scene.objects:
    if obj.name not in loaded_object_names and obj.type == "MESH":
        bpy.data.objects.remove(obj, do_unlink=True)

bproc.camera.set_intrinsics_from_blender_params(
    lens=args.camera_fov * math.pi / 180.0,
    lens_unit="FOV",
    image_height=args.image_height,
    image_width=args.image_width,
)

cameras = json.load(open(os.path.join(args.output_dir, "cameras.json"), "r"))

for cam_T in cameras["extrinsics"]:
    bproc.camera.add_camera_pose(cam_T)

# Replace the rendering section with:
if not args.render_rgb:
    """Fast depth-only rendering using Eevee"""
    # Switch to EEVEE and configure for fast depth rendering
    bpy.context.scene.render.engine = "BLENDER_EEVEE"
    bpy.context.view_layer.use_pass_z = True

    # Minimize render settings for speed
    bpy.context.scene.eevee.taa_render_samples = 1
    bpy.context.scene.eevee.use_ssr = False
    bpy.context.scene.eevee.use_ssr_refraction = False
    bpy.context.scene.eevee.use_gtao = False
    bpy.context.scene.eevee.use_bloom = False
    bpy.context.scene.eevee.use_motion_blur = False
else:
    # Keep using Cycles for high quality RGB rendering
    bproc.renderer.set_light_bounces(
        diffuse_bounces=200,
        glossy_bounces=200,
        max_bounces=200,
        transmission_bounces=200,
        transparent_max_bounces=200,
    )

    if args.cctextures_dir is not None:
        # Only apply textures if RGB rendering is enabled
        cc_materials = bproc.loader.load_ccmaterials(
            args.cctextures_dir, ["Bricks", "Wood", "Carpet", "Tile", "Marble"]
        )

        floors = bproc.filter.by_attr(loaded_objects, "name", "Floor.*", regex=True)
        for floor in floors:
            # For each material of the object
            for i in range(len(floor.get_materials())):
                # In 95% of all cases
                if np.random.uniform(0, 1) <= 0.95:
                    # Replace the material with a random one
                    floor.set_material(i, random.choice(cc_materials))

        baseboards_and_doors = bproc.filter.by_attr(
            loaded_objects, "name", "Baseboard.*|Door.*", regex=True
        )
        wood_floor_materials = bproc.filter.by_cp(
            cc_materials, "asset_name", "WoodFloor.*", regex=True
        )
        for obj in baseboards_and_doors:
            # For each material of the object
            for i in range(len(obj.get_materials())):
                # Replace the material with a random one
                obj.set_material(i, random.choice(wood_floor_materials))

        walls = bproc.filter.by_attr(loaded_objects, "name", "Wall.*", regex=True)
        marble_materials = bproc.filter.by_cp(
            cc_materials, "asset_name", "Marble.*", regex=True
        )
        for wall in walls:
            # For each material of the object
            for i in range(len(wall.get_materials())):
                # In 50% of all cases
                if np.random.uniform(0, 1) <= 0.1:
                    # Replace the material with a random one
                    wall.set_material(i, random.choice(marble_materials))

    bproc.renderer.enable_segmentation_output(map_by=["category_id"])
bproc.renderer.enable_depth_output(activate_antialiasing=False)

# Full rendering pipeline for RGB or segmentation
data = bproc.renderer.render()

# Save the data to the output directory
for key, values in data.items():
    if key == "colors" and args.render_rgb:
        for i, value in enumerate(values):
            imageio.imwrite(f"{args.output_dir}/image_{i:04d}.png", value)  # [H, W, 3]

    if key == "depth":
        for i, value in enumerate(values):
            value[value > 1e4] = 0.0  # 65504.0 or 1e10
            np.save(
                f"{args.output_dir}/depth_{i:04d}.npy", value.astype(np.float32)
            )  # [H, W]

    if key == "category_id_segmaps":
        for i, value in enumerate(values):
            assert (
                value.min() >= 0 and value.max() <= 255
            ), "Segmentation map should be in the range of [0, 255] or save to npy"
            np.save(f"{args.output_dir}/{key}_{i:04d}.npy", value.astype(np.uint8))
