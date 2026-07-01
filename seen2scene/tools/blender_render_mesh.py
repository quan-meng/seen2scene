import argparse
import json
import math
import os
from typing import Any, Dict, List, Optional

import bpy
import mathutils
import numpy as np


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def import_mesh(mesh_path: str) -> List[bpy.types.Object]:
    before = set(obj.name for obj in bpy.context.scene.objects)
    ext = os.path.splitext(mesh_path)[1].lower()

    if ext == ".ply":
        try:
            ret = bpy.ops.wm.ply_import(filepath=mesh_path)
            if ret != {'FINISHED'}:
                raise RuntimeError(f"wm.ply_import returned {ret}")
        except Exception:
            bpy.ops.import_mesh.ply(filepath=mesh_path)
    elif ext == ".obj":
        try:
            bpy.ops.wm.obj_import(filepath=mesh_path)
        except Exception:
            bpy.ops.import_scene.obj(filepath=mesh_path)
    else:
        raise ValueError(f"Unsupported mesh format: {mesh_path}")

    imported = [
        obj
        for obj in bpy.context.scene.objects
        if obj.name not in before and obj.type == "MESH"
    ]
    if len(imported) == 0:
        raise RuntimeError(f"No mesh objects imported from {mesh_path}")
    return imported


def create_material(
    default_color: tuple, default_roughness: float, style: Optional[Dict[str, Any]] = None
):
    style = style or {}
    color = tuple(style.get("color", default_color))
    roughness = float(style.get("roughness", default_roughness))
    alpha = float(style.get("alpha", 1.0))
    emission_strength = float(style.get("emission_strength", 0.0))

    mat = bpy.data.materials.new(name="MeshMaterial")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    output = nodes.new(type="ShaderNodeOutputMaterial")

    if bool(style.get("volume_fill", False)):
        # Solid semi-transparent volume: transparent surface + volume scatter.
        # Volume Scatter fills the mesh interior with colored scattered light,
        # giving a solid tinted-glass block appearance.
        transparent = nodes.new(type="ShaderNodeBsdfTransparent")
        links.new(transparent.outputs["BSDF"], output.inputs["Surface"])
        vol = nodes.new(type="ShaderNodeVolumeScatter")
        vol.inputs["Color"].default_value = color
        vol.inputs["Density"].default_value = float(style.get("volume_density", 3.0))
        links.new(vol.outputs["Volume"], output.inputs["Volume"])
    else:
        if emission_strength > 0.0:
            shader = nodes.new(type="ShaderNodeEmission")
            shader.inputs["Color"].default_value = color
            shader.inputs["Strength"].default_value = emission_strength
            shaded_output = shader.outputs[0]
        else:
            bsdf = nodes.new(type="ShaderNodeBsdfPrincipled")
            bsdf.inputs["Base Color"].default_value = color
            bsdf.inputs["Roughness"].default_value = roughness
            bsdf.inputs["Metallic"].default_value = float(style.get("metallic", 0.0))
            bsdf.inputs["IOR"].default_value = float(style.get("ior", 1.5))
            if "Specular IOR Level" in bsdf.inputs:
                bsdf.inputs["Specular IOR Level"].default_value = float(
                    style.get("specular_ior_level", 0.5)
                )
            shaded_output = bsdf.outputs["BSDF"]

        if alpha < 0.999:
            transparent = nodes.new(type="ShaderNodeBsdfTransparent")
            mix = nodes.new(type="ShaderNodeMixShader")
            mix.inputs["Fac"].default_value = 1.0 - alpha
            links.new(transparent.outputs["BSDF"], mix.inputs[1])
            links.new(shaded_output, mix.inputs[2])
            links.new(mix.outputs["Shader"], output.inputs["Surface"])
        else:
            links.new(shaded_output, output.inputs["Surface"])

        if bool(style.get("backface_culling", False)):
            mat.use_backface_culling = True

    return mat


def apply_style_to_objects(
    objects: List[bpy.types.Object],
    default_color: tuple,
    default_roughness: float,
    style: Optional[Dict[str, Any]] = None,
):
    style = style or {}
    if bool(style.get("use_vertex_color", False)):
        return
    material = create_material(default_color, default_roughness, style)
    for obj in objects:
        # Clear all existing materials so PLY-imported materials don't interfere
        obj.data.materials.clear()
        obj.data.materials.append(material)

        # Remove any vertex color attributes from PLY import
        if obj.data.color_attributes:
            for attr in list(obj.data.color_attributes):
                obj.data.color_attributes.remove(attr)

        # Shading mode: flat (faceted) or smooth
        if style.get("shade_flat", False):
            obj.data.polygons.foreach_set("use_smooth", [False] * len(obj.data.polygons))
        else:
            obj.data.polygons.foreach_set("use_smooth", [True] * len(obj.data.polygons))

        if style.get("wireframe", False):
            mod = obj.modifiers.new("Wireframe", "WIREFRAME")
            mod.thickness = float(style.get("wire_thickness", 0.003))
            mod.use_replace = bool(style.get("wire_replace", False))


def set_camera_pose(eye: np.ndarray, target: np.ndarray):
    f = eye - target
    f = f / (np.linalg.norm(f) + 1e-8)
    up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    r = np.cross(up, f)
    if np.linalg.norm(r) < 1e-6:
        up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        r = np.cross(up, f)
    r = r / (np.linalg.norm(r) + 1e-8)
    u = np.cross(f, r)
    u = u / (np.linalg.norm(u) + 1e-8)

    M = np.eye(4, dtype=np.float32)
    M[:3, 0] = r
    M[:3, 1] = u
    M[:3, 2] = f
    M[:3, 3] = eye

    bpy.ops.object.camera_add(location=eye.tolist())
    cam = bpy.context.object
    cam.data.type = "PERSP"
    cam.data.angle = math.pi / 2.0  # match pyrender yfov=np.pi/2
    cam.matrix_world = mathutils.Matrix(M.tolist())
    bpy.context.scene.camera = cam


def setup_lighting(intensity: float, sun_angle: float = 0.02, use_fill: bool = True):
    # Key light — sun from above
    bpy.ops.object.light_add(type="SUN", location=(0.0, 0.0, 30.0))
    sun = bpy.context.object
    sun.data.energy = float(intensity)
    sun.data.angle = float(sun_angle)

    if use_fill:
        # Fill light — small area, much weaker than key (4:1 ratio)
        bpy.ops.object.light_add(type="AREA", location=(-4.0, -4.0, 4.0))
        area = bpy.context.object
        area.data.energy = float(intensity) * 0.25
        area.data.size = 2.0


def render_one(
    mesh_paths: List[str],
    out_path: str,
    eye: np.ndarray,
    target: np.ndarray,
    resolution,
    num_samples: int,
    light_intensity: float,
    material_color,
    material_roughness: float,
    mesh_styles: Optional[List[Dict[str, Any]]] = None,
    world_color=(1.0, 1.0, 1.0, 1.0),
    world_strength: float = 0.15,
    sun_angle: float = 0.02,
    use_fill_light: bool = True,
    view_transform: str = "",
    use_denoising: bool = False,
    max_bounces: int = 12,
):
    clear_scene()
    mesh_styles = mesh_styles or []
    for j, mesh_path in enumerate(mesh_paths):
        meshes = import_mesh(mesh_path)
        style = mesh_styles[j] if j < len(mesh_styles) else {}
        apply_style_to_objects(
            meshes,
            default_color=material_color,
            default_roughness=material_roughness,
            style=style,
        )

    set_camera_pose(eye, target)
    setup_lighting(light_intensity, sun_angle=sun_angle, use_fill=use_fill_light)

    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = int(num_samples)
    scene.render.resolution_x = int(resolution[0])
    scene.render.resolution_y = int(resolution[1])
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.filepath = out_path
    scene.render.film_transparent = True

    # Cycles quality settings
    scene.cycles.max_bounces = max_bounces
    scene.cycles.use_denoising = use_denoising

    # Color management
    if view_transform:
        scene.view_settings.view_transform = view_transform

    # Enable GPU rendering if available (OPTIX for RTX/H100, fallback CUDA, then CPU)
    try:
        prefs = bpy.context.preferences.addons["cycles"].preferences
        for device_type in ("OPTIX", "CUDA", "HIP"):
            try:
                prefs.compute_device_type = device_type
                prefs.get_devices()
                gpu_devices = [d for d in prefs.devices if d.type != "CPU"]
                if gpu_devices:
                    for d in prefs.devices:
                        d.use = d.type != "CPU"
                    scene.cycles.device = "GPU"
                    break
            except Exception:
                continue
    except Exception:
        pass  # No cycles addon prefs — stay on CPU

    world = scene.world if scene.world is not None else bpy.data.worlds.new("World")
    scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg is not None:
        bg.inputs["Color"].default_value = tuple(world_color)
        bg.inputs["Strength"].default_value = float(world_strength)

    bpy.ops.render.render(write_still=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = json.load(f)

    if "scene_mesh_paths" in cfg:
        scene_mesh_paths = cfg["scene_mesh_paths"]
    else:
        scene_mesh_paths = [[p] for p in cfg["mesh_paths"]]
    mesh_styles = cfg.get("mesh_styles", None)
    num_views = int(cfg["num_views"])
    resolution = cfg["resolution"]
    centers = np.asarray(cfg["centers"], dtype=np.float32)
    radii = np.asarray(cfg["radii"], dtype=np.float32)
    theta = float(cfg["theta"])
    theta_rad = math.radians(theta)
    azimuth = float(cfg.get("azimuth", 180.0))
    azimuth_rad = math.radians(azimuth)
    light_intensity = float(cfg["light_intensity"])
    num_samples = int(cfg.get("num_samples", 128))
    output_dir = cfg["output_dir"]
    material_color = tuple(cfg.get("material_color", [0.604, 0.643, 0.686, 1.0]))
    material_roughness = float(cfg.get("material_roughness", 0.5))
    world_color = tuple(cfg.get("world_color", [1.0, 1.0, 1.0, 1.0]))
    world_strength = float(cfg.get("world_strength", 0.15))
    sun_angle = float(cfg.get("sun_angle", 0.02))
    use_fill_light = bool(cfg.get("use_fill_light", True))
    view_transform = str(cfg.get("view_transform", ""))
    use_denoising = bool(cfg.get("use_denoising", False))
    max_bounces = int(cfg.get("max_bounces", 12))

    camera_eyes: Optional[np.ndarray] = None
    camera_targets: Optional[np.ndarray] = None
    if "camera_eyes" in cfg and "camera_targets" in cfg:
        camera_eyes = np.asarray(cfg["camera_eyes"], dtype=np.float32)
        camera_targets = np.asarray(cfg["camera_targets"], dtype=np.float32)

    angles = (
        np.linspace(0.0, 2.0 * math.pi, num_views + 1, dtype=np.float32)[:-1]
        + azimuth_rad
    )

    for i, mesh_paths in enumerate(scene_mesh_paths):
        mesh_styles_i = None
        if mesh_styles is not None and i < len(mesh_styles):
            mesh_styles_i = mesh_styles[i]
        for v in range(num_views):
            if camera_eyes is not None:
                eye = camera_eyes[i][v]
                target = camera_targets[i][v]
            else:
                center = centers[i]
                radius = radii[i]
                eye = np.array(
                    [
                        center[0]
                        + radius * math.cos(float(angles[v])) * math.cos(theta_rad),
                        center[1]
                        + radius * math.sin(float(angles[v])) * math.cos(theta_rad),
                        center[2] + radius * math.sin(theta_rad),
                    ],
                    dtype=np.float32,
                )
                target = center

            out_path = os.path.join(output_dir, f"mesh_{i:04d}_view_{v:04d}.png")
            render_one(
                mesh_paths=mesh_paths,
                out_path=out_path,
                eye=eye,
                target=target,
                resolution=resolution,
                num_samples=num_samples,
                light_intensity=light_intensity,
                material_color=material_color,
                material_roughness=material_roughness,
                mesh_styles=mesh_styles_i,
                world_color=world_color,
                world_strength=world_strength,
                sun_angle=sun_angle,
                use_fill_light=use_fill_light,
                view_transform=view_transform,
                use_denoising=use_denoising,
                max_bounces=max_bounces,
            )


if __name__ == "__main__":
    main()
