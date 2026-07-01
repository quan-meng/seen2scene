"""
Unified rendering module supporting both pyrender and blender backends.

Usage:
    from seen2scene.tools.render import mesh2images, RenderBackend, PAPER_MATERIAL_COLOR

    # Use pyrender (fast, for development)
    images = mesh2images(meshes, backend=RenderBackend.PYRENDER, ...)

    # Use blender (high quality, for paper figures)
    images = mesh2images(meshes, backend=RenderBackend.BLENDER, ...)
"""

import os
import math
import json
import tempfile
import subprocess
from pathlib import Path
from enum import Enum
from typing import *
import numpy as np
import trimesh
from einops import rearrange

from .common_utils import clip_mesh


def _export_ply_blender_compat(mesh: trimesh.Trimesh, path: str) -> None:
    """Export mesh as ASCII PLY compatible with Blender 4.0+.

    Blender 4.0's binary PLY importer has a bug that rejects certain meshes
    with "Invalid face size".  ASCII format reliably works across all versions.
    """
    verts = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    header = (
        "ply\n"
        "format ascii 1.0\n"
        f"element vertex {len(verts)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        f"element face {len(faces)}\n"
        "property list uchar int vertex_indices\n"
        "end_header\n"
    )
    with open(path, "w") as f:
        f.write(header)
        # Write vertices using numpy for speed
        np.savetxt(f, verts, fmt="%.7g")
        # Write faces: prepend vertex count (3) to each row
        face_data = np.column_stack([np.full(len(faces), 3, dtype=np.int32), faces])
        np.savetxt(f, face_data, fmt="%d")


class RenderBackend(str, Enum):
    PYRENDER = "pyrender"
    BLENDER = "blender"


# Default material color for research paper rendering — Warm Clay
# Matched from lt3sd template.blend (metallic=0.5, IOR=1.0, Specular IOR Level=0.6)
PAPER_MATERIAL_COLOR = (0.820, 0.620, 0.480, 1.0)  # RGBA — Warm Clay #D19E7A
PAPER_MATERIAL_ROUGHNESS = 0.5

# lt3sd-matched Blender rendering recipe
PAPER_LIGHT_INTENSITY = 1.0
PAPER_WORLD_STRENGTH = 0.8
PAPER_SUN_ANGLE = 1.047  # 60° soft shadow
PAPER_USE_FILL_LIGHT = False
PAPER_VIEW_TRANSFORM = "Standard"
PAPER_USE_DENOISING = True
PAPER_MAX_BOUNCES = 6
# Per-mesh style dict for the standard paper look
PAPER_MESH_STYLE = {
    "color": list(PAPER_MATERIAL_COLOR),
    "shade_flat": True,
    "roughness": 0.5,
    "metallic": 0.5,
    "ior": 1.0,
    "specular_ior_level": 0.6,
}


def _look_at(
    eye: np.ndarray, target: np.ndarray, up: Tuple[float, float, float] = (0, 0, 1)
) -> np.ndarray:
    """Compute camera-to-world look-at matrix matching pyrender conventions."""
    eye = np.asarray(eye, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    up_vec = np.asarray(up, dtype=np.float32)

    forward = eye - target
    forward_norm = np.linalg.norm(forward)
    if forward_norm < 1e-6:
        return np.eye(4, dtype=np.float32)
    forward = forward / forward_norm
    right = np.cross(up_vec, forward)
    right_norm = np.linalg.norm(right)
    if right_norm < 1e-6:
        up_vec = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        right = np.cross(up_vec, forward)
        right_norm = np.linalg.norm(right)
        if right_norm < 1e-6:
            return np.eye(4, dtype=np.float32)
    right = right / right_norm
    true_up = np.cross(forward, right)
    true_up = true_up / (np.linalg.norm(true_up) + 1e-8)

    pose = np.eye(4, dtype=np.float32)
    pose[:3, 0] = right
    pose[:3, 1] = true_up
    pose[:3, 2] = forward
    pose[:3, 3] = eye
    return pose


def look_at(
    eye: np.ndarray,  # shape: [N, 3], camera location
    target: np.ndarray,  # shape: [N, 3], target position
    up: np.ndarray = np.array([0, 0, 1]),  # shape: [3], up vector
    system: str = "opengl",  # camera coordinate system: "blender", "opencv", or "opengl"
) -> np.ndarray:  # returns: [N, 4, 4] camera-to-world transformation matrix
    """Compute batch-wise look-at transformation matrices.

    Args:
        eye: Camera locations of shape [N, 3]
        target: Target positions of shape [N, 3]
        up: Up vector of shape [3], defaults to [0, 0, 1]
        system: Camera coordinate system, one of:
            - "blender": RIGHT, UP, BACK
            - "opencv": RIGHT, DOWN, FRONT
            - "opengl": RIGHT, UP, BACK (default)

    Returns:
        World-to-camera transformation matrices of shape [N, 4, 4]
    """
    # Compute the forward vector from target to eye
    f = eye - target
    f /= np.linalg.norm(f, axis=1, keepdims=True)

    # Compute the right vector
    r = np.cross(up, f)
    r /= np.linalg.norm(r, axis=1, keepdims=True)

    # Recompute the up vector
    u = np.cross(f, r)
    u /= np.linalg.norm(u, axis=1, keepdims=True)

    # Create a 4x4 look-at matrix
    lookat_matrix = np.eye(4)[None].repeat(eye.shape[0], axis=0)
    lookat_matrix[:, :3, 0] = r
    lookat_matrix[:, :3, 1] = u
    lookat_matrix[:, :3, 2] = f
    lookat_matrix[:, :3, 3] = eye

    # Adjust for different camera systems
    if system.lower() == "opencv":
        # OpenCV: RIGHT, DOWN, FRONT
        lookat_matrix[:, 1:3, :3] *= -1
    elif system.lower() == "opengl":
        # OpenGL: RIGHT, UP, BACK
        pass
    else:
        raise ValueError(f"Unknown camera system: {system}")

    return lookat_matrix


def spherical_camera_pose(
    radius: float = 1.0,
    up_vector: tuple = (0.0, 1, 0),
    center=np.array((0.0, 0, 0)),
    theta: float = 60,
    num_views: int = 20,
    azimuth: float = 0.0,
):
    """
    Create camera poses looking at the center of the sphere.
    """
    theta = math.radians(theta)
    angles = np.linspace(0, 2 * np.pi, num_views + 1)[:-1] + math.radians(azimuth)

    x = center[0] + radius * np.cos(angles) * math.cos(theta)
    y = center[1] + radius * np.sin(angles) * math.cos(theta)
    z = center[2] + radius * np.ones((num_views,)) * math.sin(theta)

    eye = np.stack([x, y, z], axis=1)  # [num_views, 3]

    poses = look_at(eye, center, up_vector)

    return poses


def _compute_bbox_centers_radii(
    bbox_world: np.ndarray,
    ceiling_clip: Optional[float],
    ratio: float = 1.0,
    yfov: float = np.pi / 2.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    bbox_world = np.asarray(bbox_world, dtype=np.float32)
    if bbox_world.ndim == 1:
        bbox_world = bbox_world[None]

    if ceiling_clip is not None:
        bbox_world = bbox_world.copy()
        bbox_world[..., -1] = bbox_world[..., 2] + ceiling_clip

    centers = (bbox_world[:, :3] + bbox_world[:, 3:]) / 2
    half_diag = np.linalg.norm(bbox_world[:, 3:] - bbox_world[:, :3], axis=1) / 2.0
    radii = half_diag / (ratio * math.tan(yfov / 2.0))
    return bbox_world, centers, radii


def _prepare_meshes(
    mesh_list: List[trimesh.Trimesh],
    bbox_world: np.ndarray,
    ceiling_clip: Optional[float],
    ratio: float = 1.0,
) -> Tuple[List[trimesh.Trimesh], np.ndarray, np.ndarray, np.ndarray]:
    if len(mesh_list) == 0:
        return [], np.empty((0, 6), dtype=np.float32), np.empty((0, 3)), np.empty((0,))

    bbox_world, centers, radii = _compute_bbox_centers_radii(bbox_world, ceiling_clip, ratio=ratio)
    if len(mesh_list) != len(bbox_world):
        raise ValueError(
            f"mesh_list length {len(mesh_list)} != bbox_world length {len(bbox_world)}"
        )

    clipped_meshes = [
        clip_mesh(mesh, bbox_world_i)
        for mesh, bbox_world_i in zip(mesh_list, bbox_world)
    ]
    return clipped_meshes, bbox_world, centers, radii


def _render_pyrender(
    mesh_list: List[trimesh.Trimesh],
    num_views: int,
    resolution: Union[int, Tuple[int, int]],
    bbox_world: np.ndarray,
    ceiling_clip: Optional[float],
    theta: float,
    azimuth: float,
    light_intensity: float,
    composite_background: bool = True,
    ratio: float = 1.0,
) -> np.ndarray:
    """Render using pyrender backend."""
    from seen2scene.tools.pyrender_utils import render_mesh, PerspectiveCamera

    camera = PerspectiveCamera(yfov=np.pi / 2.0)

    mesh_list, bbox_world, centers, radii = _prepare_meshes(
        mesh_list, bbox_world, ceiling_clip, ratio=ratio
    )
    images = []
    for i, mesh in enumerate(mesh_list):
        poses = spherical_camera_pose(
            num_views=num_views,
            center=centers[i],
            radius=radii[i],
            theta=theta,
            azimuth=azimuth,
            up_vector=(0, 0, 1),
        )

        view_images = []
        for pose in poses:
            color, depth = render_mesh(
                mesh, camera, pose, resolution=resolution, intensity=light_intensity
            )
            if not composite_background:
                # Add alpha channel from depth buffer
                alpha = (depth > 0).astype(color.dtype)[..., None]
                color = np.concatenate([color, alpha], axis=-1)
            view_images.append(color)
        images.append(view_images)

    images = np.array(images)  # [B, V, H, W, C]
    images = rearrange(images, "b v h w c -> b v c h w")
    return images


def _render_pyrender_poses(
    mesh_list: List[trimesh.Trimesh],
    poses_list: List[List[np.ndarray]],
    resolution: Union[int, Tuple[int, int]],
    light_intensity: float,
    ambient_light: np.ndarray = None,
    extra_light_poses: list = None,
) -> np.ndarray:
    """Render pyrender using explicit camera poses."""
    from seen2scene.tools.pyrender_utils import render_mesh, PerspectiveCamera

    camera = PerspectiveCamera(yfov=np.pi / 2.0)

    if len(mesh_list) != len(poses_list):
        raise ValueError(
            f"mesh_list length {len(mesh_list)} != poses_list length {len(poses_list)}"
        )
    view_counts = {len(poses) for poses in poses_list}
    if len(view_counts) > 1:
        raise ValueError(f"Inconsistent view counts in poses_list: {view_counts}")

    images = []
    for mesh, poses in zip(mesh_list, poses_list):
        view_images = [
            render_mesh(
                mesh, camera, pose, resolution=resolution, intensity=light_intensity,
                ambient_light=ambient_light, extra_light_poses=extra_light_poses,
            )[0]
            for pose in poses
        ]
        images.append(view_images)

    images = np.array(images)  # [B, V, H, W, C]
    images = rearrange(images, "b v h w c -> b v c h w")
    return images


def _render_blender(
    mesh_list: List[Union[trimesh.Trimesh, List[trimesh.Trimesh]]],
    num_views: int,
    resolution: Union[int, Tuple[int, int]],
    bbox_world: np.ndarray,
    ceiling_clip: Optional[float],
    theta: float,
    azimuth: float,
    light_intensity: float,
    blender_env: str = "blender",
    num_samples: int = 128,
    material_color: Tuple[float, float, float, float] = PAPER_MATERIAL_COLOR,
    material_roughness: float = PAPER_MATERIAL_ROUGHNESS,
    camera_eyes: Optional[List[np.ndarray]] = None,
    camera_targets: Optional[List[np.ndarray]] = None,
    mesh_styles: Optional[List[Union[Dict[str, Any], List[Dict[str, Any]]]]] = None,
    composite_background: bool = True,
    ratio: float = 1.0,
    world_color: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
    world_strength: float = PAPER_WORLD_STRENGTH,
    sun_angle: float = PAPER_SUN_ANGLE,
    use_fill_light: bool = PAPER_USE_FILL_LIGHT,
    view_transform: str = PAPER_VIEW_TRANSFORM,
    use_denoising: bool = PAPER_USE_DENOISING,
    max_bounces: int = PAPER_MAX_BOUNCES,
) -> np.ndarray:
    """Render using blender backend via subprocess."""
    import cv2

    if isinstance(resolution, int):
        resolution = (resolution, resolution)

    bbox_world, centers, radii = _compute_bbox_centers_radii(bbox_world, ceiling_clip, ratio=ratio)
    scene_meshes: List[List[trimesh.Trimesh]] = []
    for m in mesh_list:
        if isinstance(m, (list, tuple)):
            scene_meshes.append(list(m))
        else:
            scene_meshes.append([m])

    if len(scene_meshes) != len(bbox_world):
        raise ValueError(
            f"mesh_list length {len(scene_meshes)} != bbox_world length {len(bbox_world)}"
        )

    clipped_scene_meshes: List[List[trimesh.Trimesh]] = []
    for i, meshes_i in enumerate(scene_meshes):
        clipped_scene_meshes.append([clip_mesh(mesh, bbox_world[i]) for mesh in meshes_i])
    if camera_eyes is not None and camera_targets is None:
        raise ValueError("camera_targets must be provided with camera_eyes")
    if camera_targets is not None and camera_eyes is None:
        raise ValueError("camera_eyes must be provided with camera_targets")
    if camera_eyes is not None:
        if len(camera_eyes) != len(scene_meshes):
            raise ValueError(
                f"camera_eyes length {len(camera_eyes)} != mesh_list length {len(scene_meshes)}"
            )
        if len(camera_targets) != len(scene_meshes):
            raise ValueError(
                f"camera_targets length {len(camera_targets)} != mesh_list length {len(scene_meshes)}"
            )
        view_counts = {len(eyes) for eyes in camera_eyes}
        if len(view_counts) > 1:
            raise ValueError(f"Inconsistent view counts in camera_eyes: {view_counts}")
        num_views = next(iter(view_counts)) if view_counts else 0
    elif centers is None or radii is None:
        raise ValueError(
            "centers and radii are required when camera_eyes is not provided"
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        # Clip and save meshes, skipping empty ones (e.g. clipped to nothing)
        scene_mesh_paths: List[List[str]] = []
        filtered_styles: Optional[List] = [] if mesh_styles is not None else None
        for i, meshes_i in enumerate(clipped_scene_meshes):
            mesh_paths_i = []
            styles_i = (mesh_styles[i] if i < len(mesh_styles) else None) if mesh_styles is not None else None
            kept_styles_i: Optional[List] = [] if isinstance(styles_i, list) else None
            for j, mesh in enumerate(meshes_i):
                if len(mesh.vertices) == 0:
                    continue  # skip empty meshes (e.g. clipped to nothing)
                # Rebuild as clean triangulated mesh to ensure Blender-compatible PLY
                # (Blender rejects faces with >255 vertices)
                mesh = trimesh.Trimesh(
                    vertices=mesh.vertices, faces=mesh.faces, process=True
                )
                if len(mesh.faces) == 0:
                    continue
                mesh_path = os.path.join(tmpdir, f"mesh_{i:04d}_{j:03d}.ply")
                _export_ply_blender_compat(mesh, mesh_path)
                mesh_paths_i.append(mesh_path)
                if kept_styles_i is not None:
                    kept_styles_i.append(styles_i[j] if j < len(styles_i) else {})
            scene_mesh_paths.append(mesh_paths_i)
            if filtered_styles is not None:
                filtered_styles.append(kept_styles_i if kept_styles_i is not None else styles_i)

        # Prepare config for blender subprocess
        # Note: mesh is already clipped, so we pass pre-computed centers/radii
        config = {
            "scene_mesh_paths": scene_mesh_paths,
            "num_views": num_views,
            "resolution": resolution,
            "centers": centers.tolist(),
            "radii": radii.tolist(),
            "theta": theta,
            "azimuth": azimuth,
            "light_intensity": light_intensity,
            "num_samples": num_samples,
            "output_dir": tmpdir,
            "material_color": list(material_color),
            "material_roughness": material_roughness,
            "world_color": list(world_color),
            "world_strength": world_strength,
            "sun_angle": sun_angle,
            "use_fill_light": use_fill_light,
            "view_transform": view_transform,
            "use_denoising": use_denoising,
            "max_bounces": max_bounces,
        }
        if filtered_styles is not None:
            config["mesh_styles"] = filtered_styles
        if camera_eyes is not None:
            config["camera_eyes"] = [np.asarray(eyes).tolist() for eyes in camera_eyes]
            config["camera_targets"] = [
                np.asarray(targets).tolist() for targets in camera_targets
            ]
        config_path = os.path.join(tmpdir, "render_config.json")
        with open(config_path, "w") as f:
            json.dump(config, f)

        # Get the path to the blender render script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        render_script = os.path.join(
            script_dir, "blender_render_mesh.py"
        )

        # Run blender subprocess
        cmd = [
            "conda",
            "run",
            "-n",
            blender_env,
            "python",
            render_script,
            "--config",
            config_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0 and (
            "EnvironmentLocationNotFound" in result.stderr
            or "command not found" in result.stderr
            or "No such file or directory" in result.stderr
            or result.stderr == ""
        ):
            import sys as _sys
            cmd_direct = [_sys.executable, render_script, "--config", config_path]
            env = os.environ.copy()
            conda_lib = str(Path(_sys.executable).parent.parent / "lib")
            env["LD_LIBRARY_PATH"] = conda_lib + ":" + env.get("LD_LIBRARY_PATH", "")
            result = subprocess.run(cmd_direct, capture_output=True, text=True, env=env)

        if result.returncode != 0:
            raise RuntimeError(
                f"Blender rendering failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
            )

        # Load rendered images
        images = []
        for i in range(len(scene_meshes)):
            view_images = []
            for v in range(num_views):
                img_path = os.path.join(tmpdir, f"mesh_{i:04d}_view_{v:04d}.png")
                if not os.path.exists(img_path):
                    raise FileNotFoundError(f"Rendered image not found: {img_path}")
                img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
                if img is None:
                    raise ValueError(f"Failed to read image: {img_path}")
                # Handle RGBA or RGB
                if img.shape[-1] == 4:
                    if composite_background:
                        alpha = img[..., 3:4].astype(np.float32) / 255.0
                        rgb = img[..., :3].astype(np.float32)
                        white_bg = np.ones_like(rgb) * 255.0
                        img = (rgb * alpha + white_bg * (1 - alpha)).astype(np.uint8)
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    else:
                        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
                else:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # Resize if needed
                if img.shape[:2] != resolution:
                    img = cv2.resize(img, resolution)
                view_images.append(img.astype(np.float32) / 255.0)
            images.append(view_images)

        images = np.array(images)  # [B, V, H, W, C]
        images = rearrange(images, "b v h w c -> b v c h w")

    return images


def mesh2images(
    mesh_list: Union[List[trimesh.Trimesh], List[List[trimesh.Trimesh]]],
    num_views: int = 1,
    resolution: Union[int, Tuple[int, int]] = 150,
    bbox_world: np.ndarray = np.array([[-1.0, -1.0, -1.0, 1.0, 1.0, 1.0]]),
    ceiling_clip: Optional[float] = None,
    theta: float = 60.0,
    light_intensity: float = 5.0,
    backend: Union[RenderBackend, str] = RenderBackend.PYRENDER,
    pose_type: Literal["spherical", "lookdown"] = "spherical",
    # Blender-specific options
    blender_env: str = "blender",
    num_samples: int = 128,
    material_color: Tuple[float, float, float, float] = PAPER_MATERIAL_COLOR,
    material_roughness: float = PAPER_MATERIAL_ROUGHNESS,
    mesh_styles: Optional[List[Union[Dict[str, Any], List[Dict[str, Any]]]]] = None,
    composite_background: bool = True,
    azimuth: float = 0.0,
    ratio: float = 1.0,
    world_color: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
    world_strength: float = PAPER_WORLD_STRENGTH,
    sun_angle: float = PAPER_SUN_ANGLE,
    use_fill_light: bool = PAPER_USE_FILL_LIGHT,
    view_transform: str = PAPER_VIEW_TRANSFORM,
    use_denoising: bool = PAPER_USE_DENOISING,
    max_bounces: int = PAPER_MAX_BOUNCES,
) -> np.ndarray:
    """
    Render meshes to images with unified backend support.

    Args:
        mesh_list: List of trimesh objects to render
        num_views: Number of views per mesh
        resolution: Image resolution (height, width) or single int for square
        bbox_world: Bounding boxes [B, 6] as [xmin, ymin, zmin, xmax, ymax, zmax]
        ceiling_clip: Optional ceiling clipping height
        theta: Camera elevation angle in degrees
        light_intensity: Light intensity (pyrender) or sun strength (blender)
        backend: Rendering backend - "pyrender" or "blender"
        pose_type: Camera pose type (only "spherical" supported currently)
        blender_env: Conda environment name for blender (only used with blender backend)
        num_samples: Number of samples for blender rendering
        material_color: RGBA color for blender material (default: PAPER_MATERIAL_COLOR #9AA4AF,
            a neutral blue-gray clay color commonly used in research paper figures)
        material_roughness: Roughness for blender material (0=shiny, 1=matte, default: 0.5)
        ratio: How much the rendered mesh fills the frame (1.0=tight, 0.75≈old default)

    Returns:
        images: numpy array of shape [B, V, C, H, W] with values in [0, 1]
    """
    if isinstance(backend, str):
        backend = RenderBackend(backend)

    if pose_type != "spherical":
        raise ValueError(f"Only 'spherical' pose_type is supported, got: {pose_type}")

    if backend == RenderBackend.PYRENDER:
        return _render_pyrender(
            mesh_list=mesh_list,
            num_views=num_views,
            resolution=resolution,
            bbox_world=bbox_world,
            ceiling_clip=ceiling_clip,
            theta=theta,
            azimuth=azimuth,
            light_intensity=light_intensity,
            composite_background=composite_background,
            ratio=ratio,
        )
    elif backend == RenderBackend.BLENDER:
        return _render_blender(
            mesh_list=mesh_list,
            num_views=num_views,
            resolution=resolution,
            bbox_world=bbox_world,
            ceiling_clip=ceiling_clip,
            theta=theta,
            azimuth=azimuth,
            light_intensity=light_intensity,
            blender_env=blender_env,
            num_samples=num_samples,
            material_color=material_color,
            material_roughness=material_roughness,
            mesh_styles=mesh_styles,
            composite_background=composite_background,
            ratio=ratio,
            world_color=world_color,
            world_strength=world_strength,
            sun_angle=sun_angle,
            use_fill_light=use_fill_light,
            view_transform=view_transform,
            use_denoising=use_denoising,
            max_bounces=max_bounces,
        )
    else:
        raise ValueError(f"Unknown backend: {backend}")


def mesh2video(
    mesh_list: Union[List[trimesh.Trimesh], List[List[trimesh.Trimesh]]],
    num_frames: int = 120,
    resolution: Union[int, Tuple[int, int]] = 150,
    bbox_world: np.ndarray = np.array([[-1.0, -1.0, -1.0, 1.0, 1.0, 1.0]]),
    ceiling_clip: Optional[float] = None,
    theta: float = 60.0,
    light_intensity: float = 5.0,
    backend: Union[RenderBackend, str] = RenderBackend.PYRENDER,
    trajectory: Literal["orbit", "object_zoom", "custom"] = "orbit",
    object_bboxes: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
    zoom_frames: int = 12,
    hold_frames: int = 6,
    zoom_scale: float = 0.7,
    object_order: Literal["volume", "input", "center"] = "volume",
    camera_eyes: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
    camera_targets: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
    output_path: Optional[Union[str, List[str]]] = None,
    fps: int = 24,
    blender_env: str = "blender",
    num_samples: int = 128,
    material_color: Tuple[float, float, float, float] = PAPER_MATERIAL_COLOR,
    material_roughness: float = PAPER_MATERIAL_ROUGHNESS,
    ratio: float = 1.0,
    ambient_light: np.ndarray = None,
    extra_light_poses: list = None,
) -> np.ndarray:
    """
    Render meshes to a video sequence with trajectory control.

    Notes:
        object_bboxes are expected in world coordinates.

    Returns:
        images: numpy array of shape [B, T, C, H, W] with values in [0, 1]
    """
    if isinstance(backend, str):
        backend = RenderBackend(backend)

    def _expand_paths(
        path: Optional[Union[str, List[str]]], batch_size: int
    ) -> List[Optional[str]]:
        if path is None:
            return [None] * batch_size
        if isinstance(path, (list, tuple)):
            if len(path) != batch_size:
                raise ValueError(
                    f"output_path length {len(path)} != batch size {batch_size}"
                )
            return list(path)
        if batch_size == 1:
            return [path]
        base, ext = os.path.splitext(path)
        if ext == "":
            ext = ".mp4"
        return [f"{base}_{i:04d}{ext}" for i in range(batch_size)]

    def _write_video(frames: np.ndarray, path: str, fps_value: int) -> None:
        import cv2

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        frames = np.clip(frames, 0.0, 1.0)
        frames = (frames * 255.0).astype(np.uint8)
        height, width = frames.shape[-2], frames.shape[-1]
        writer = cv2.VideoWriter(
            path, cv2.VideoWriter_fourcc(*"mp4v"), fps_value, (width, height)
        )
        for frame in frames:
            frame_bgr = cv2.cvtColor(frame.transpose(1, 2, 0), cv2.COLOR_RGB2BGR)
            writer.write(frame_bgr)
        writer.release()

    if trajectory == "orbit":
        images = mesh2images(
            mesh_list=mesh_list,
            num_views=num_frames,
            resolution=resolution,
            bbox_world=bbox_world,
            ceiling_clip=ceiling_clip,
            theta=theta,
            light_intensity=light_intensity,
            backend=backend,
            pose_type="spherical",
            blender_env=blender_env,
            num_samples=num_samples,
            material_color=material_color,
            material_roughness=material_roughness,
            ratio=ratio,
        )
    else:
        mesh_list = list(mesh_list)
        bbox_world, centers, radii = _compute_bbox_centers_radii(
            bbox_world, ceiling_clip, ratio=ratio
        )
        if len(mesh_list) != len(bbox_world):
            raise ValueError(
                f"mesh_list length {len(mesh_list)} != bbox_world length {len(bbox_world)}"
            )

        if trajectory == "custom":
            if camera_eyes is None or camera_targets is None:
                raise ValueError(
                    "camera_eyes and camera_targets are required for custom"
                )

            if isinstance(camera_eyes, np.ndarray):
                if (
                    camera_eyes.ndim == 3
                    and len(mesh_list) > 1
                    and camera_eyes.shape[0] == len(mesh_list)
                ):
                    camera_eyes = list(camera_eyes)
                else:
                    camera_eyes = [camera_eyes]
            if isinstance(camera_targets, np.ndarray):
                if (
                    camera_targets.ndim == 3
                    and len(mesh_list) > 1
                    and camera_targets.shape[0] == len(mesh_list)
                ):
                    camera_targets = list(camera_targets)
                else:
                    camera_targets = [camera_targets]
        elif trajectory == "object_zoom":
            if object_bboxes is None:
                raise ValueError("object_bboxes are required for object_zoom")

            if isinstance(object_bboxes, np.ndarray):
                object_bboxes = [object_bboxes]
            elif len(object_bboxes) != len(mesh_list):
                object_bboxes_arr = np.asarray(object_bboxes)
                if len(mesh_list) != 1 or object_bboxes_arr.ndim not in (2, 3):
                    raise ValueError(
                        "object_bboxes must be a list per mesh or a single array for batch size 1"
                    )
                object_bboxes = [object_bboxes_arr]

            camera_eyes = []
            camera_targets = []
            for idx, bboxes in enumerate(object_bboxes):
                if hasattr(bboxes, "cpu"):
                    bboxes = bboxes.cpu().numpy()
                bboxes = np.asarray(bboxes, dtype=np.float32)
                if bboxes.size == 0:
                    raise ValueError("object_bboxes is empty for object_zoom")
                if bboxes.ndim == 2 and bboxes.shape[-1] == 6:
                    mins = bboxes[:, :3]
                    maxs = bboxes[:, 3:]
                elif bboxes.ndim == 3 and bboxes.shape[-2:] == (2, 3):
                    mins = bboxes[:, 0]
                    maxs = bboxes[:, 1]
                else:
                    raise ValueError(f"Unsupported object_bboxes shape: {bboxes.shape}")

                centers_obj = (mins + maxs) / 2
                sizes_obj = maxs - mins
                volumes = np.prod(sizes_obj, axis=1)
                order = np.arange(len(centers_obj))
                if object_order == "volume":
                    order = np.argsort(-volumes)
                elif object_order == "center":
                    dists = np.linalg.norm(
                        centers_obj[:, :2] - centers[idx][None, :2], axis=1
                    )
                    order = np.argsort(dists)
                elif object_order != "input":
                    raise ValueError(f"Unknown object_order: {object_order}")

                centers_obj = centers_obj[order]
                sizes_obj = sizes_obj[order]

                start_radius = max(float(radii[idx]), 1e-3)
                start_eye = np.array(
                    [centers[idx][0], centers[idx][1], centers[idx][2] + start_radius],
                    dtype=np.float32,
                )

                start_target = centers[idx].astype(np.float32)
                per_object_eyes = [start_eye]
                per_object_targets = [start_target]
                prev_target = start_target
                for center_obj, size_obj in zip(centers_obj, sizes_obj):
                    obj_radius = np.linalg.norm(size_obj) / 1.5
                    near_radius = max(obj_radius * zoom_scale, 1e-3)
                    center_obj = center_obj.astype(np.float32)
                    direction = start_eye - center_obj
                    dist_start = np.linalg.norm(direction)
                    if dist_start < 1e-6:
                        direction = np.array([0.0, 0.0, 1.0], dtype=np.float32)
                        dist_start = 1.0
                    else:
                        direction = direction / dist_start
                    near_radius = min(near_radius, max(dist_start * 0.95, 1e-3))
                    obj_eye = center_obj + direction * near_radius

                    zoom_in = (
                        np.linspace(0.0, 1.0, max(zoom_frames, 1) + 1, endpoint=False)[
                            1:
                        ]
                        if zoom_frames > 0
                        else []
                    )
                    zoom_out = (
                        np.linspace(1.0, 0.0, max(zoom_frames, 1) + 1, endpoint=True)[
                            1:
                        ]
                        if zoom_frames > 0
                        else []
                    )

                    if hold_frames > 0:
                        transition = np.linspace(
                            0.0, 1.0, hold_frames + 1, endpoint=True
                        )[1:]
                        for t in transition:
                            target = (1 - t) * prev_target + t * center_obj
                            per_object_eyes.append(start_eye)
                            per_object_targets.append(target.astype(np.float32))

                    for t in zoom_in:
                        eye = (1 - t) * start_eye + t * obj_eye
                        per_object_eyes.append(eye.astype(np.float32))
                        per_object_targets.append(center_obj)

                    for _ in range(hold_frames):
                        per_object_eyes.append(obj_eye)
                        per_object_targets.append(center_obj)

                    for t in zoom_out:
                        eye = (1 - t) * start_eye + t * obj_eye
                        per_object_eyes.append(eye.astype(np.float32))
                        per_object_targets.append(center_obj)

                    prev_target = center_obj

                camera_eyes.append(np.stack(per_object_eyes, axis=0))
                camera_targets.append(np.stack(per_object_targets, axis=0))
        else:
            raise ValueError(f"Unknown trajectory: {trajectory}")

        if camera_eyes is None or camera_targets is None:
            raise ValueError("camera_eyes and camera_targets must be defined")

        if len(camera_eyes) != len(mesh_list) or len(camera_targets) != len(mesh_list):
            raise ValueError("camera_eyes/targets must match batch size")
        view_counts = {len(eyes) for eyes in camera_eyes}
        if len(view_counts) > 1:
            raise ValueError(f"Inconsistent view counts in camera_eyes: {view_counts}")

        if backend == RenderBackend.PYRENDER:
            mesh_list_clipped, _, _, _ = _prepare_meshes(
                mesh_list, bbox_world, ceiling_clip
            )
            poses_list = [
                [_look_at(eye, target) for eye, target in zip(eyes, targets)]
                for eyes, targets in zip(camera_eyes, camera_targets)
            ]
            images = _render_pyrender_poses(
                mesh_list=mesh_list_clipped,
                poses_list=poses_list,
                resolution=resolution,
                light_intensity=light_intensity,
                ambient_light=ambient_light,
                extra_light_poses=extra_light_poses,
            )
        elif backend == RenderBackend.BLENDER:
            images = _render_blender(
                mesh_list=mesh_list,
                num_views=next(iter(view_counts)) if view_counts else 0,
                resolution=resolution,
                bbox_world=bbox_world,
                ceiling_clip=ceiling_clip,
                theta=theta,
                light_intensity=light_intensity,
                blender_env=blender_env,
                num_samples=num_samples,
                material_color=material_color,
                material_roughness=material_roughness,
                camera_eyes=camera_eyes,
                camera_targets=camera_targets,
            )
        else:
            raise ValueError(f"Unknown backend: {backend}")

    output_paths = _expand_paths(output_path, images.shape[0])
    for frames, path in zip(images, output_paths):
        if path is not None:
            _write_video(frames, path, fps)

    return images
