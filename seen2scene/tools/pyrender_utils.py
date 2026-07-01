from typing import Union, Tuple
import dataclasses
import os
import trimesh
import numpy as np
import pyrender
from pyrender import (
    DirectionalLight,
    SpotLight,
    PointLight,
    OffscreenRenderer,
    RenderFlags,
)

os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["MUJOCO_GL"] = "egl"
from .log_utils import get_logger

logger = get_logger(file_name=__file__)


def opencv_to_pyrender(pose):
    """
    Convert OpenCV camera extrinsics (R, T) to PyRender's camera pose matrix.

    :param R: 3x3 Rotation matrix (OpenCV, world to camera)
    :param T: 3x1 Translation vector (OpenCV, world to camera)
    :return: 4x4 Transformation matrix (PyRender, camera to world)
    """
    R = pose[:3, :3]
    T = pose[:3, 3]

    # Invert the rotation and translation for camera to world conversion
    R_inv = R.T
    T_inv = -np.dot(R_inv, T)

    # Create a 4x4 transformation matrix for PyRender
    pose = np.eye(4)
    pose[:3, :3] = R_inv
    pose[:3, 3] = T_inv.ravel()

    # Adjust for PyRender's coordinate system (flip the Z axis)
    flip_z = np.eye(4)
    flip_z[2, 2] = -1  # Flip the Z-axis
    pose = np.dot(pose, flip_z)

    return pose


def init_light(scene, intensity=6.0, pose=None, extra_poses=None) -> None:
    direc_l = DirectionalLight(color=np.ones(3), intensity=intensity)
    spot_l = SpotLight(
        color=np.ones(3),
        intensity=intensity,
        innerConeAngle=np.pi / 16,
        outerConeAngle=np.pi / 6,
    )
    point_l = PointLight(color=np.ones(3), intensity=2 * intensity)
    scene.add(direc_l, pose=pose)
    scene.add(point_l, pose=pose)
    scene.add(spot_l, pose=pose)
    # Additional fill lights at other poses (e.g., top-down) to reduce dark areas
    if extra_poses is not None:
        fill_intensity = intensity * 0.5
        for ep in extra_poses:
            scene.add(DirectionalLight(color=np.ones(3), intensity=fill_intensity), pose=ep)


def reorder_faces_for_camera(
    mesh: trimesh.Trimesh, camera_position: np.ndarray
) -> trimesh.Trimesh:
    """
    Reorder faces in a mesh to prevent back-face culling based on camera position.

    Args:
        mesh: trimesh object
        camera_position: [3] array, camera position in world coordinates

    Returns:
        trimesh.Trimesh: New mesh with reordered faces
    """
    vertices = mesh.vertices
    faces = mesh.faces.copy()  # Create a copy to avoid modifying original mesh

    # Calculate face centers
    face_centers = vertices[faces].mean(axis=1)

    # Calculate vectors from camera to face centers
    camera_to_face = face_centers - camera_position

    # Calculate face normals
    face_normals = np.cross(
        vertices[faces[:, 1]] - vertices[faces[:, 0]],
        vertices[faces[:, 2]] - vertices[faces[:, 0]],
    )

    # Add small epsilon to avoid division by zero
    norms = np.linalg.norm(face_normals, axis=1, keepdims=True)
    eps = 1e-10
    norms = np.maximum(norms, eps)  # Ensure no zeros in denominator
    face_normals = face_normals / norms

    # Calculate dot product between camera rays and face normals
    dots = np.sum(camera_to_face * face_normals, axis=1)

    # Flip faces where dot product is positive (facing away from camera)
    flip_mask = dots > 0
    faces[flip_mask] = faces[flip_mask][:, ::-1]

    # Collect all visual attributes
    visual_kwargs = {}
    if mesh.visual.kind == "face":
        face_colors = mesh.visual.face_colors.copy()
        if flip_mask.any():
            face_colors[flip_mask] = face_colors[flip_mask]  # No need to reverse colors
        visual_kwargs["face_colors"] = face_colors
    elif mesh.visual.kind == "vertex":
        visual_kwargs["vertex_colors"] = mesh.visual.vertex_colors

    # Create new mesh with reordered faces and preserved colors
    return trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        process=False,  # Prevent trimesh from processing/changing the mesh
        visual=trimesh.visual.ColorVisuals(**visual_kwargs),
    )


@dataclasses.dataclass
class OrthographicCamera:
    target: str = "pyrender.OrthographicCamera"
    xmag: float = 1.0
    ymag: float = 1.0
    znear: float = 0.05
    zfar: float = 20.0


@dataclasses.dataclass
class PerspectiveCamera:
    target: str = "pyrender.PerspectiveCamera"
    yfov: float = np.pi / 2.0
    aspectRatio: float = 1.0
    znear: float = 0.05
    zfar: float = 20.0


@dataclasses.dataclass
class IntrinsicsCamera:
    target: str = "pyrender.IntrinsicsCamera"
    fx: float = 1.0
    fy: float = 1.0
    cx: float = 0.0
    cy: float = 0.0
    znear: float = 0.05
    zfar: float = 20.0


def render_mesh(
    mesh: Union[trimesh.Trimesh, str],
    camera: Union[OrthographicCamera, PerspectiveCamera, IntrinsicsCamera],
    camera_pose,
    resolution: Union[int, Tuple[int, int]] = 300,
    light: bool = True,
    intensity: float = 5.0,
    light_pose: np.ndarray = None,
    bg_color=None,
    only_depth: bool = False,
    ambient_light: np.ndarray = None,
    extra_light_poses: list = None,
):
    """
    Render a mesh with a given camera pose.
    Args:
        mesh: trimesh object
        camera_pose: 4x4 camera to world matrix
        resolution: int
        light: bool
        intensity: float
        bg_color: None or [3]
        only_depth: bool, optimize for depth-only rendering
    Return:
        - color: [H, W, 3], float, [0, 1] (None if only_depth=True)
        - depth: [H, W], float, [0, ~]
    """
    # renderer
    resolution = (resolution, resolution) if isinstance(resolution, int) else resolution

    if isinstance(mesh, str):
        mesh = trimesh.load(mesh, process=False)

    if len(mesh.vertices) == 0:
        if light and not only_depth:
            return np.ones([*resolution, 3]), np.zeros(resolution)
        else:
            return np.zeros(resolution)

    # Reorder faces to prevent back-face culling
    camera_position = camera_pose[:3, 3]
    mesh = reorder_faces_for_camera(mesh, camera_position)

    # Convert to pyrender mesh
    mesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)

    r = OffscreenRenderer(resolution[0], resolution[1])
    scene = pyrender.Scene(bg_color=bg_color, ambient_light=ambient_light)
    scene.add(mesh)

    camera_params = dataclasses.asdict(camera)
    module, name = camera_params.pop("target").rsplit(".", 1)
    camera = getattr(__import__(module, fromlist=[name]), name)(**camera_params)
    scene.add(camera, pose=camera_pose)

    if light and not only_depth:
        light_pose = camera_pose if light_pose is None else light_pose
        init_light(scene, intensity=intensity, pose=light_pose,
                   extra_poses=extra_light_poses)

        render_flags = RenderFlags.ALL_SOLID | RenderFlags.FACE_NORMALS
        color, depth = r.render(scene, flags=render_flags)
        r.delete()
        color = color.astype(np.float32) / 255.0
        return color, depth
    else:
        depth = r.render(scene, flags=RenderFlags.DEPTH_ONLY)
        r.delete()
        return depth
