#!/usr/bin/env python3
"""Export a lightweight .blend template with the rendering environment.

Run with:
    blender --background --python assets/export_blend_template.py

Creates assets/render_template.blend matching template.blend settings:
  - Cycles engine (GPU, 256 samples, denoising on, transparent film)
  - Sun light (energy=1.0, angle=60°)
  - World background (white, strength 0.8)
  - Color management: Standard view transform
  - Camera (perspective, ~39.6° FOV)
  - "MeshMaterial" (Principled BSDF) ready to assign
  - No mesh objects — import your own in Blender
"""

import math
import bpy
import mathutils
import numpy as np


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    # Remove orphan data
    for block in bpy.data.meshes:
        bpy.data.meshes.remove(block)
    for block in bpy.data.materials:
        bpy.data.materials.remove(block)


def setup_camera(eye, target, fov=0.6911):
    """Create a perspective camera."""
    eye = np.asarray(eye, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)

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
    cam.name = "Camera"
    cam.data.type = "PERSP"
    cam.data.angle = fov
    cam.matrix_world = mathutils.Matrix(M.tolist())
    bpy.context.scene.camera = cam


def setup_lighting(intensity=1.0, sun_angle=math.radians(60.0)):
    """Sun light matching template.blend (energy=1.0, angle=60°, no fill)."""
    bpy.ops.object.light_add(type="SUN", location=(0.0, 0.0, 30.0))
    sun = bpy.context.object
    sun.name = "Sun"
    sun.data.energy = float(intensity)
    sun.data.angle = float(sun_angle)


def setup_world(color=(1.0, 1.0, 1.0, 1.0), strength=0.8):
    """White world background (strength=0.8 matching template.blend)."""
    scene = bpy.context.scene
    world = scene.world if scene.world else bpy.data.worlds.new("World")
    scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg is not None:
        bg.inputs["Color"].default_value = tuple(color)
        bg.inputs["Strength"].default_value = float(strength)


def setup_render(resolution=(2048, 2048), num_samples=256, max_bounces=6):
    """Cycles render settings matching template.blend."""
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = num_samples
    scene.render.resolution_x = resolution[0]
    scene.render.resolution_y = resolution[1]
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.film_transparent = True
    scene.cycles.max_bounces = max_bounces
    scene.cycles.use_denoising = True

    # Color management: Standard (not AgX)
    scene.view_settings.view_transform = "Standard"

    # Enable GPU rendering if available
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
        pass


def setup_material(
    color=(0.820, 0.620, 0.480, 1.0),  # Warm Clay #D19E7A (PAPER_MATERIAL_COLOR)
    roughness=0.5,
):
    """Create a Principled BSDF material.

    The material is saved in the .blend file as "MeshMaterial".
    After importing a mesh, assign it via:
        mesh_obj.data.materials.append(bpy.data.materials["MeshMaterial"])
    """
    mat = bpy.data.materials.new(name="MeshMaterial")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    output = nodes.new(type="ShaderNodeOutputMaterial")
    bsdf = nodes.new(type="ShaderNodeBsdfPrincipled")
    bsdf.inputs["Base Color"].default_value = color
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Metallic"].default_value = 0.0
    bsdf.inputs["IOR"].default_value = 1.5
    if "Specular IOR Level" in bsdf.inputs:
        bsdf.inputs["Specular IOR Level"].default_value = 0.5
    links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])

    # Mark as fake user so it persists without being assigned to any object
    mat.use_fake_user = True


def main():
    import os

    clear_scene()

    # Camera matching template.blend: looking at origin from (-3, 0, 6)
    eye = np.array([-3.0, 0.0, 6.0], dtype=np.float32)
    target = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    setup_camera(eye, target, fov=0.6911)  # ~39.6° FOV

    setup_lighting(intensity=1.0, sun_angle=math.radians(60.0))
    setup_world(color=(1.0, 1.0, 1.0, 1.0), strength=0.8)
    setup_render(resolution=(2048, 2048), num_samples=256, max_bounces=6)
    setup_material()

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "render_template.blend")
    bpy.ops.wm.save_as_mainfile(filepath=out_path, compress=True)
    print(f"Saved template: {out_path}")


if __name__ == "__main__":
    main()
