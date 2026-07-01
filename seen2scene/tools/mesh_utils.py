import numpy as np
import os
import trimesh
from typing import *
import torch
from skimage import measure
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from matplotlib.colors import Colormap
from trimesh.creation import cylinder

from .log_utils import get_logger
from .render_utils import RenderBackend

logger = get_logger(file_name=__file__)


def volume2mesh(
    volume: Union[np.ndarray, torch.Tensor],
    isovalue: float = 0.0,
    bbox_min: Optional[torch.Tensor] = None,
    voxel_size: Optional[float] = None,
) -> trimesh.Trimesh:
    assert volume.ndim == 3, f"Input volume of shape {volume.shape} is wrong!"

    if isinstance(bbox_min, torch.Tensor):
        bbox_min = bbox_min.cpu().numpy()

    if not isinstance(volume, np.ndarray):
        volume = volume.cpu().numpy()

    vertices, triangles, _, _ = measure.marching_cubes(
        volume, level=isovalue, gradient_direction="descent"
    )

    if voxel_size is None:
        voxel_size = 2.0 / (volume.shape[-1] - 1)

    vertices = vertices * voxel_size + bbox_min
    mesh = trimesh.Trimesh(vertices, triangles, process=False)

    return mesh


def create_camera_primitive(
    scale=1.0,
    length=0.1,
    line_radius=0.005,
    color=[0.5, 0.5, 0.5],
    color_up=[0, 1, 0],
    color_lookat=[0.5, 0.5, 0.5],
    camera_model: str = "opencv",
):
    """
    Create a camera mesh primitive including a square pyramid and lines representing
    the up vector and lookat direction. All in the camera's local coordinate space.

    Args:
        scale: size of the camera pyramid.
        length: length of the axis lines.
        line_radius: radius of the axis lines (for thinner lines).
        color: RGB color of the camera.
        color_up: RGB color of the up vector.
        color_lookat: RGB color of the lookat direction.

    Returns:
        trimesh object for the camera primitive.
    """
    # Define pyramid vertices in local camera space
    apex = np.array([0, 0, 0])  # Camera center (apex of pyramid)
    base = scale * np.array(
        [
            [-0.05, -0.05, 0.1],  # Bottom-left corner of the base
            [0.05, -0.05, 0.1],  # Bottom-right corner of the base
            [0.05, 0.05, 0.1],  # Top-right corner of the base
            [-0.05, 0.05, 0.1],  # Top-left corner of the base
        ]
    )  # Image plane (base of the pyramid)

    lookat_vector = np.array([0, 0, 1])
    if camera_model == "opencv":
        base = base * np.array([1, 1, -1])
        lookat_vector = lookat_vector * np.array([1, 1, -1])

    # Combine vertices for the pyramid
    vertices = np.vstack([apex, base])

    # Create 6 triangular faces: 4 sides + 2 triangles for the base
    faces = [
        [0, 1, 2],  # Side 1 (apex, base bottom-left, base bottom-right)
        [0, 2, 3],  # Side 2 (apex, base bottom-right, base top-right)
        [0, 3, 4],  # Side 3 (apex, base top-right, base top-left)
        [0, 4, 1],  # Side 4 (apex, base top-left, base bottom-left)
        [1, 2, 3],  # Base triangle 1 (bottom-left, bottom-right, top-right)
        [1, 3, 4],  # Base triangle 2 (bottom-left, top-right, top-left)
    ]

    # Create trimesh mesh for the pyramid
    camera_mesh = trimesh.Trimesh(
        vertices=vertices, faces=faces, vertex_colors=[color] * len(faces)
    )

    # Calculate center and top edge of the base
    center_base = np.mean(base, axis=0)
    # top_edge_base = np.mean(base[2:], axis=0)

    # Define the up vector (Y-axis) and look-at vector (Z-axis) in local space
    # up_vector = np.array([0, 1, 0])

    # # Create up-vector cylinder (line starts at the top edge of the base)
    # up_end = top_edge_base + up_vector * length
    # up_cylinder = cylinder(radius=line_radius, height=length, sections=8)
    # up_cylinder.apply_translation(top_edge_base + (up_end - top_edge_base) / 2)
    # up_cylinder.visual.vertex_colors = color_up

    # Create look-at vector cylinder (line starts at the center of the base)
    lookat_end = center_base + lookat_vector * length * scale
    lookat_cylinder = cylinder(radius=line_radius * scale, height=length, sections=8)
    lookat_cylinder.apply_translation(center_base + (lookat_end - center_base) / 2)
    lookat_cylinder.visual.vertex_colors = color_lookat

    # Combine the pyramid and the two lines into a single mesh
    return trimesh.util.concatenate([camera_mesh, lookat_cylinder])


def create_transformed_camera(
    camera_pose, scale=1.0, length=0.2, line_radius=0.005, color=[1, 0, 0]
):
    """
    Create a transformed camera by applying the camera pose to the primitive.

    Args:
        camera_pose: [4x4] pose matrix for the camera.
        scale: size of the camera pyramid.
        length: length of the axis lines.
        line_radius: radius of the axis lines (for thinner lines).
        color: RGB color of the camera.

    Returns:
        Transformed camera mesh (pyramid + axes).
    """
    # Create the camera primitive in local space
    camera_primitive = create_camera_primitive(
        scale=scale, length=length, line_radius=line_radius, color=color
    )

    # Apply the camera pose (rotation and translation)
    camera_primitive.apply_transform(camera_pose)

    return camera_primitive


def cameras2mesh(
    camera_poses,
    export_path: str = "camera_scene.obj",
    camera_scale: float = 1.0,
    fov: float = 60.0,
):
    """
    Visualize a list of camera poses and export to an OBJ or PLY file.

    Args:
        camera_poses: list of [4x4] camera pose matrices.
        export_path: output file path (OBJ or PLY).
    """
    scene = trimesh.Scene()

    # Loop through each camera pose
    for i, pose in enumerate(camera_poses):
        # Create a random color for each camera
        color = np.random.rand(3)

        # Create the camera mesh with the pose applied
        transformed_camera = create_transformed_camera(
            pose, scale=camera_scale, length=0.1, line_radius=0.005, color=color
        )
        scene.add_geometry(transformed_camera)

    # Export the scene to the specified file
    scene.export(export_path)


def bboxes2mesh(
    bboxes: Union[np.ndarray, torch.Tensor],
    categories: Optional[Union[np.ndarray, torch.Tensor]] = None,
    palette: Optional[Colormap] = None,
    style: Literal["box", "wireframe"] = "box",
    wire_thickness: float = 0.02,
):
    """
    Docstring for bboxes2mesh

    bboxes: [N, 2, 3] or torch.Tensor
    categories: [N] or torch.Tensor
    palette: Colormap for coloring the boxes
    style: "box" for solid boxes or "wireframe" for edge-only boxes
    wire_thickness: thickness for wireframe edges (world units)
    """
    if isinstance(bboxes, torch.Tensor):
        bboxes = bboxes.cpu().numpy()

    meshes = []
    for j, bbox in enumerate(bboxes):
        size = bbox[1] - bbox[0]
        if size.min() > 0.0:
            center = (bbox[1] + bbox[0]) / 2
            if style == "box":
                box = trimesh.creation.box(extents=size)
                box.apply_translation(center)
                mesh = box
            elif style == "wireframe":
                thickness = max(float(wire_thickness), 1e-6)
                edges = [
                    ([0, 0, 0], [1, 0, 0]),
                    ([1, 0, 0], [1, 1, 0]),
                    ([1, 1, 0], [0, 1, 0]),
                    ([0, 1, 0], [0, 0, 0]),
                    ([0, 0, 1], [1, 0, 1]),
                    ([1, 0, 1], [1, 1, 1]),
                    ([1, 1, 1], [0, 1, 1]),
                    ([0, 1, 1], [0, 0, 1]),
                    ([0, 0, 0], [0, 0, 1]),
                    ([1, 0, 0], [1, 0, 1]),
                    ([1, 1, 0], [1, 1, 1]),
                    ([0, 1, 0], [0, 1, 1]),
                ]
                edge_meshes = []
                for a, b in edges:
                    start = bbox[0] + size * np.array(a, dtype=np.float32)
                    end = bbox[0] + size * np.array(b, dtype=np.float32)
                    direction = end - start
                    length = np.linalg.norm(direction)
                    if length < 1e-6:
                        continue
                    mid = (start + end) / 2
                    edge_box = trimesh.creation.box(
                        extents=[thickness, thickness, length]
                    )
                    z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float32)
                    direction = direction / length
                    axis = np.cross(z_axis, direction)
                    axis_norm = np.linalg.norm(axis)
                    if axis_norm < 1e-6:
                        if np.dot(z_axis, direction) > 0:
                            rot = np.eye(3, dtype=np.float32)
                        else:
                            rot = trimesh.transformations.rotation_matrix(
                                np.pi, [1, 0, 0]
                            )[:3, :3]
                    else:
                        axis = axis / axis_norm
                        angle = np.arccos(np.clip(np.dot(z_axis, direction), -1.0, 1.0))
                        rot = trimesh.transformations.rotation_matrix(angle, axis)[
                            :3, :3
                        ]
                    transform = np.eye(4, dtype=np.float32)
                    transform[:3, :3] = rot
                    transform[:3, 3] = mid
                    edge_box.apply_transform(transform)
                    edge_meshes.append(edge_box)
                if edge_meshes:
                    mesh = trimesh.util.concatenate(edge_meshes)
                else:
                    mesh = trimesh.Trimesh(
                        vertices=np.empty((0, 3)), faces=np.empty((0, 3))
                    )
            else:
                raise ValueError(f"Unknown style: {style}")
            if categories is not None and palette is not None:
                cat_idx = int(categories[j])
                color = palette[cat_idx]
                mesh.visual.vertex_colors = np.repeat(
                    color[None, :], repeats=mesh.vertices.shape[0], axis=0
                )

            meshes.append(mesh)

    if len(meshes) > 0:
        mesh = trimesh.util.concatenate(meshes)
    else:
        mesh = trimesh.Trimesh(vertices=np.empty((0, 3)), faces=np.empty((0, 3)))

    return mesh


def mesh2pclouds(
    meshes: Union[List[trimesh.Trimesh], List[List[trimesh.Trimesh]]],
    num_points: int = 2048,
):
    pclouds = []
    for tgt in meshes:
        try:
            points = trimesh.sample.sample_surface(tgt, num_points)[0]
        except Exception as e:
            points = np.empty((0, 3))
        pclouds.append(torch.from_numpy(points))  # [N, 3]

    return torch.stack(pclouds)  # [B, N, 3]


def mesh2images(
    mesh_list: Union[List[trimesh.Trimesh], List[List[trimesh.Trimesh]]],
    num_views: int = 1,
    resolution: Union[int, Tuple[int, int]] = 150,
    bbox_world: np.ndarray = np.array([[-1.0, -1.0, -1.0, 1.0, 1.0, 1.0]]),
    ceiling_clip: Optional[float] = None,
    theta: float = 60.0,
    light_intensity: float = 5.0,
    pose_type: Literal["spherical", "lookdown"] = "spherical",
    backend: Union[RenderBackend, str] = RenderBackend.PYRENDER,
    blender_env: str = "blender",
    num_samples: int = 128,
    material_color: Optional[Tuple[float, float, float, float]] = None,
    material_roughness: Optional[float] = None,
    mesh_styles: Optional[List[Union[Dict[str, Any], List[Dict[str, Any]]]]] = None,
    composite_background: bool = True,
    azimuth: float = 0.0,
    ratio: float = 1.0,
    world_color: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
    world_strength: float = 0.8,
    sun_angle: float = 1.047,
    use_fill_light: bool = False,
    view_transform: str = "Standard",
    use_denoising: bool = True,
    max_bounces: int = 6,
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
        light_intensity: Light intensity
        pose_type: Camera pose type (only "spherical" supported)
        backend: Rendering backend - "pyrender" (fast) or "blender" (high quality)
        blender_env: Conda environment name for blender (only used with blender backend)
        num_samples: Number of samples for blender rendering
        ratio: How much the rendered mesh fills the frame (1.0=tight, 0.75≈old default)
        world_color: RGBA color for Blender world background (ambient tint)
        world_strength: Strength of Blender world background lighting

    Returns:
        images: numpy array of shape [B, V, C, H, W] with values in [0, 1]
    """
    from seen2scene.tools.render_utils import (
        mesh2images as render_mesh2images,
        PAPER_MATERIAL_COLOR,
        PAPER_MATERIAL_ROUGHNESS,
    )

    if material_color is None:
        material_color = PAPER_MATERIAL_COLOR
    if material_roughness is None:
        material_roughness = PAPER_MATERIAL_ROUGHNESS

    return render_mesh2images(
        mesh_list=mesh_list,
        num_views=num_views,
        resolution=resolution,
        bbox_world=bbox_world,
        ceiling_clip=ceiling_clip,
        theta=theta,
        light_intensity=light_intensity,
        backend=backend,
        pose_type=pose_type,
        blender_env=blender_env,
        num_samples=num_samples,
        material_color=material_color,
        material_roughness=material_roughness,
        mesh_styles=mesh_styles,
        composite_background=composite_background,
        azimuth=azimuth,
        ratio=ratio,
        world_color=world_color,
        world_strength=world_strength,
        sun_angle=sun_angle,
        use_fill_light=use_fill_light,
        view_transform=view_transform,
        use_denoising=use_denoising,
        max_bounces=max_bounces,
    )


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
    material_color: Tuple[float, float, float, float] = None,
    material_roughness: float = None,
    ratio: float = 1.0,
) -> np.ndarray:
    from seen2scene.tools.render_utils import (
        mesh2video as render_mesh2video,
        PAPER_MATERIAL_COLOR,
        PAPER_MATERIAL_ROUGHNESS,
    )

    if material_color is None:
        material_color = PAPER_MATERIAL_COLOR
    if material_roughness is None:
        material_roughness = PAPER_MATERIAL_ROUGHNESS

    return render_mesh2video(
        mesh_list=mesh_list,
        num_frames=num_frames,
        resolution=resolution,
        bbox_world=bbox_world,
        ceiling_clip=ceiling_clip,
        theta=theta,
        light_intensity=light_intensity,
        backend=backend,
        trajectory=trajectory,
        object_bboxes=object_bboxes,
        zoom_frames=zoom_frames,
        hold_frames=hold_frames,
        zoom_scale=zoom_scale,
        object_order=object_order,
        camera_eyes=camera_eyes,
        camera_targets=camera_targets,
        output_path=output_path,
        fps=fps,
        blender_env=blender_env,
        num_samples=num_samples,
        material_color=material_color,
        material_roughness=material_roughness,
        ratio=ratio,
    )


def detect_floor_z(
    mesh: trimesh.Trimesh,
    bin_size: float = 0.02,
    min_peak_ratio: float = 0.05,
    smoothing_sigma: float = 2.0,
    outlier_percentile: float = 1.0,
    method: Literal["histogram", "density", "normal", "combined"] = "combined",
    normal_threshold: float = 0.8,
) -> float:
    """
    Robustly detect the floor z-value from a 3D mesh captured with LiDAR.

    This function handles artifact floaters below the actual floor by using
    a combination of histogram analysis, density-based filtering, and normal analysis.

    Args:
        mesh: A trimesh.Trimesh object representing the 3D scene.
        bin_size: Size of histogram bins in meters (default: 0.02m = 2cm).
        min_peak_ratio: Minimum ratio of peak height to max peak to be considered
            significant (default: 0.05 = 5%).
        smoothing_sigma: Gaussian smoothing sigma for histogram (default: 2.0).
        outlier_percentile: Percentile to filter extreme outliers (default: 1.0).
        method: Detection method - "histogram", "density", "normal", or "combined" (default).
        normal_threshold: Minimum upward normal component (nz) to consider a vertex
            as floor-facing (default: 0.8, i.e., within ~37 degrees of vertical).

    Returns:
        floor_z: The estimated z-coordinate of the floor.

    Example:
        >>> mesh = trimesh.load("scene.ply")
        >>> floor_z = detect_floor_z(mesh)
        >>> print(f"Floor is at z = {floor_z:.3f}")
    """
    # Extract z-coordinates from vertices
    z_values = mesh.vertices[:, 2].copy()

    if len(z_values) == 0:
        raise ValueError("Mesh has no vertices")

    # Step 1: Remove extreme outliers using percentiles
    z_min_clip = np.percentile(z_values, outlier_percentile)
    z_max_clip = np.percentile(z_values, 100 - outlier_percentile)
    z_filtered = z_values[(z_values >= z_min_clip) & (z_values <= z_max_clip)]

    results = {}

    if method in ["histogram", "combined"]:
        results["histogram"] = _detect_floor_histogram(
            z_filtered, bin_size, min_peak_ratio, smoothing_sigma
        )

    if method in ["density", "combined"]:
        results["density"] = _detect_floor_density(z_filtered, bin_size)

    if method in ["normal", "combined"]:
        results["normal"] = _detect_floor_by_normal(
            mesh, bin_size, min_peak_ratio, smoothing_sigma, normal_threshold
        )

    if method == "histogram":
        return results["histogram"]
    elif method == "density":
        return results["density"]
    elif method == "normal":
        return results["normal"]
    else:  # combined
        # Normal-based detection is most reliable as it filters out
        # walls/ceilings and finds the largest horizontal surface
        # Use normal method as primary, fall back to histogram if they agree
        if "normal" in results:
            normal_z = results["normal"]
            hist_z = results.get("histogram", normal_z)
            # If histogram and normal agree (within 0.5m), average them
            if abs(normal_z - hist_z) < 0.5:
                return (normal_z + hist_z) / 2
            else:
                # They disagree - trust normal method (filters by surface orientation)
                return normal_z
        else:
            # No normal available, use histogram
            return results.get("histogram", results.get("density", 0.0))


def _detect_floor_histogram(
    z_values: np.ndarray,
    bin_size: float,
    min_peak_ratio: float,
    smoothing_sigma: float,
) -> float:
    """Detect floor using histogram peak analysis."""
    # Create histogram
    z_min, z_max = z_values.min(), z_values.max()
    n_bins = max(int((z_max - z_min) / bin_size), 10)
    hist, bin_edges = np.histogram(z_values, bins=n_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Smooth histogram to reduce noise
    hist_smooth = gaussian_filter1d(hist.astype(float), sigma=smoothing_sigma)

    # Find peaks in the histogram
    # The floor should be a significant peak (many points at similar z)
    peaks, properties = find_peaks(
        hist_smooth,
        height=hist_smooth.max() * min_peak_ratio,
        distance=max(3, int(0.1 / bin_size)),  # At least 10cm apart
    )

    if len(peaks) == 0:
        # No peaks found, fall back to percentile
        return np.percentile(z_values, 5)

    # Get peak heights
    peak_heights = hist_smooth[peaks]

    # Find the lowest significant peak (likely the floor)
    # Sort peaks by z-value (ascending)
    sorted_indices = np.argsort(bin_centers[peaks])
    sorted_peaks = peaks[sorted_indices]
    sorted_heights = peak_heights[sorted_indices]

    # The floor is typically one of the lowest significant peaks
    # Look for the first peak that has substantial height
    threshold = sorted_heights.max() * 0.3  # At least 30% of max peak height

    for i, (peak_idx, height) in enumerate(zip(sorted_peaks, sorted_heights)):
        if height >= threshold:
            return bin_centers[peak_idx]

    # Fallback: return the lowest peak
    return bin_centers[sorted_peaks[0]]


def _detect_floor_by_normal(
    mesh: trimesh.Trimesh,
    bin_size: float,
    min_peak_ratio: float,
    smoothing_sigma: float,
    normal_threshold: float = 0.8,
) -> float:
    """Detect floor using vertex normals - floor vertices have upward normals.

    The floor is identified as the LARGEST peak in the z-histogram of
    upward-facing vertices, since the floor typically has the highest
    concentration of upward-facing points (floaters are sparse).
    """
    # Get vertex normals
    try:
        vertex_normals = mesh.vertex_normals
    except Exception:
        # If normals not available, fall back to histogram method
        z_values = mesh.vertices[:, 2]
        return _detect_floor_histogram(
            z_values, bin_size, min_peak_ratio, smoothing_sigma
        )

    # Floor vertices have normals pointing up: nz > threshold (positive Z)
    # This filters out walls (horizontal normals) and ceilings (downward normals)
    upward_mask = vertex_normals[:, 2] > normal_threshold

    if np.sum(upward_mask) < 100:
        # Too few upward-facing vertices, fall back to histogram
        z_values = mesh.vertices[:, 2]
        return _detect_floor_histogram(
            z_values, bin_size, min_peak_ratio, smoothing_sigma
        )

    # Get z-values of upward-facing vertices
    z_upward = mesh.vertices[upward_mask, 2]

    # The floor is the LARGEST cluster of upward-facing vertices
    # (floor has many points concentrated at same z, floaters are sparse)
    z_min, z_max = z_upward.min(), z_upward.max()
    n_bins = max(int((z_max - z_min) / bin_size), 10)
    hist, bin_edges = np.histogram(z_upward, bins=n_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Smooth histogram
    hist_smooth = gaussian_filter1d(hist.astype(float), sigma=smoothing_sigma)

    # Find peaks
    peaks, _ = find_peaks(
        hist_smooth,
        height=hist_smooth.max() * min_peak_ratio,
        distance=max(3, int(0.1 / bin_size)),
    )

    if len(peaks) == 0:
        # No peaks, return the z with maximum histogram count
        return float(bin_centers[np.argmax(hist_smooth)])

    # Return the LARGEST peak (floor has most upward-facing vertices)
    peak_heights = hist_smooth[peaks]
    largest_peak_idx = peaks[np.argmax(peak_heights)]
    return float(bin_centers[largest_peak_idx])


def _detect_floor_density(
    z_values: np.ndarray,
    bin_size: float,
    window_size: float = 0.1,
) -> float:
    """Detect floor using local density analysis."""
    # Sort z values
    z_sorted = np.sort(z_values)

    # Compute local density using a sliding window
    # Density = number of points within window_size
    n_points = len(z_sorted)
    window_points = int(n_points * 0.1)  # Use 10% of points as window

    if window_points < 10:
        window_points = min(10, n_points)

    # Find the z-range with highest density in the lower portion
    # Focus on lower 30% of z-range to find floor
    z_range = z_sorted[-1] - z_sorted[0]
    search_max = z_sorted[0] + z_range * 0.3

    best_z = z_sorted[0]
    best_density = 0

    # Slide through and find highest density region
    step = max(1, n_points // 100)
    for i in range(0, n_points - window_points, step):
        z_window = z_sorted[i : i + window_points]
        z_center = z_window[window_points // 2]

        if z_center > search_max:
            break

        # Density = points per unit z
        z_span = z_window[-1] - z_window[0]
        if z_span > 0:
            density = window_points / z_span
            if density > best_density:
                best_density = density
                best_z = z_center

    return best_z


def detect_floor_z_ransac(
    mesh: trimesh.Trimesh,
    max_iterations: int = 1000,
    distance_threshold: float = 0.02,
    min_inliers_ratio: float = 0.1,
) -> Tuple[float, np.ndarray]:
    """
    Detect floor z-value using RANSAC plane fitting.

    This method fits horizontal planes (normal ~ [0, 0, 1]) and finds
    the one with most inliers at the lowest z-level.

    Args:
        mesh: A trimesh.Trimesh object.
        max_iterations: Maximum RANSAC iterations.
        distance_threshold: Maximum distance to plane for inliers (meters).
        min_inliers_ratio: Minimum ratio of inliers to consider a valid plane.

    Returns:
        floor_z: The estimated z-coordinate of the floor.
        inlier_mask: Boolean mask of vertices belonging to the floor.
    """
    vertices = mesh.vertices
    n_points = len(vertices)
    min_inliers = int(n_points * min_inliers_ratio)

    best_floor_z = None
    best_inliers = 0
    best_mask = None

    # Pre-filter: focus on lower portion of the scene
    z_values = vertices[:, 2]
    z_threshold = np.percentile(z_values, 40)  # Lower 40%
    candidate_mask = z_values < z_threshold
    candidate_indices = np.where(candidate_mask)[0]

    if len(candidate_indices) < 3:
        # Fallback to simple percentile
        return np.percentile(z_values, 5), np.zeros(n_points, dtype=bool)

    for _ in range(max_iterations):
        # Sample 3 random points from lower region
        sample_indices = np.random.choice(candidate_indices, 3, replace=False)
        sample_points = vertices[sample_indices]

        # Fit plane through these points
        v1 = sample_points[1] - sample_points[0]
        v2 = sample_points[2] - sample_points[0]
        normal = np.cross(v1, v2)

        if np.linalg.norm(normal) < 1e-6:
            continue

        normal = normal / np.linalg.norm(normal)

        # Check if plane is roughly horizontal (normal ~ [0, 0, ±1])
        if abs(normal[2]) < 0.9:  # Not horizontal enough
            continue

        # Compute distances to plane
        d = -np.dot(normal, sample_points[0])
        distances = np.abs(np.dot(vertices, normal) + d)

        # Count inliers
        inlier_mask = distances < distance_threshold
        n_inliers = np.sum(inlier_mask)

        if n_inliers >= min_inliers:
            # Compute mean z of inliers
            floor_z = np.mean(vertices[inlier_mask, 2])

            # Prefer lower planes with more inliers
            # Score = inliers - penalty for higher z
            z_penalty = (floor_z - z_values.min()) * n_points * 0.1
            score = n_inliers - z_penalty

            if best_floor_z is None or (
                n_inliers > best_inliers * 0.8 and floor_z < best_floor_z
            ):
                best_floor_z = floor_z
                best_inliers = n_inliers
                best_mask = inlier_mask

    if best_floor_z is None:
        # Fallback
        best_floor_z = np.percentile(z_values, 5)
        best_mask = np.zeros(n_points, dtype=bool)

    return best_floor_z, best_mask


def _load_trimesh(mesh_path: str) -> trimesh.Trimesh:
    mesh_obj = trimesh.load(mesh_path, process=False)
    if isinstance(mesh_obj, trimesh.Scene):
        if len(mesh_obj.geometry) == 0:
            raise ValueError(f"No geometry in mesh scene: {mesh_path}")
        mesh = trimesh.util.concatenate(tuple(mesh_obj.geometry.values()))
    elif isinstance(mesh_obj, trimesh.Trimesh):
        mesh = mesh_obj
    else:
        raise TypeError(f"Unsupported mesh type: {type(mesh_obj)}")
    return mesh


def _submesh_from_face_mask(
    mesh: trimesh.Trimesh, face_mask: np.ndarray
) -> trimesh.Trimesh:
    face_idx = np.nonzero(face_mask)[0]
    if face_idx.size == 0:
        return trimesh.Trimesh(
            vertices=np.zeros((0, 3), dtype=np.float32),
            faces=np.zeros((0, 3), dtype=np.int64),
            process=False,
        )
    return mesh.submesh([face_idx], append=True, repair=False, process=False)


def _keep_largest_components(mesh: trimesh.Trimesh, k: int) -> trimesh.Trimesh:
    if mesh.is_empty or k <= 0:
        return mesh
    parts = mesh.split(only_watertight=False)
    if len(parts) <= k:
        return mesh
    parts = sorted(parts, key=lambda m: float(m.area), reverse=True)[:k]
    return trimesh.util.concatenate(parts)


def segment_floor_walls(
    mesh_path: str,
    floor_method: Literal[
        "combined", "histogram", "density", "normal", "ransac"
    ] = "combined",
    floor_height_tol: float = 0.05,
    floor_normal_thresh: float = 0.9,
    wall_normal_z_thresh: float = 0.2,
    wall_min_height: float = 0.1,
    wall_top_percentile: float = 99.0,
    wall_top_margin: float = 0.05,
    keep_largest_components: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Segment floor and walls from a scene mesh.

    Args:
        mesh_path: Path to the scene mesh (ply/obj/etc).
        floor_method: Floor detection method. "ransac" uses detect_floor_z_ransac,
            otherwise uses detect_floor_z.
        floor_height_tol: Max distance (meters) from detected floor_z for floor faces.
        floor_normal_thresh: Threshold on |normal_z| for floor faces.
        wall_normal_z_thresh: Max |normal_z| for wall faces (near vertical).
        wall_min_height: Min height above floor for wall faces.
        wall_top_percentile: Percentile of z for wall top clipping.
        wall_top_margin: Margin (meters) below top percentile to stop wall faces.
        keep_largest_components: Keep only the largest k connected components.

    Returns:
        A dict with:
            - "mesh": original mesh
            - "floor_z": detected floor z
            - "floor_face_mask": boolean face mask
            - "wall_face_mask": boolean face mask
            - "floor_mesh": trimesh of floor
            - "wall_mesh": trimesh of walls
    """
    mesh = _load_trimesh(mesh_path)
    if mesh.faces is None or len(mesh.faces) == 0:
        raise ValueError("Mesh has no faces; cannot segment floor/walls.")

    if floor_method == "ransac":
        floor_z, _ = detect_floor_z_ransac(mesh)
    else:
        floor_z = detect_floor_z(mesh, method=floor_method)

    face_normals = mesh.face_normals
    face_centroids = mesh.triangles_center

    floor_mask = (np.abs(face_normals[:, 2]) >= floor_normal_thresh) & (
        np.abs(face_centroids[:, 2] - floor_z) <= floor_height_tol
    )

    z_vals = mesh.vertices[:, 2]
    z_top = np.percentile(z_vals, wall_top_percentile)
    if z_top <= floor_z + wall_min_height:
        z_top = float(z_vals.max())

    wall_mask = (
        (np.abs(face_normals[:, 2]) <= wall_normal_z_thresh)
        & (face_centroids[:, 2] >= floor_z + wall_min_height)
        & (face_centroids[:, 2] <= z_top - wall_top_margin)
    )
    wall_mask &= ~floor_mask

    floor_mesh = _submesh_from_face_mask(mesh, floor_mask)
    wall_mesh = _submesh_from_face_mask(mesh, wall_mask)

    if keep_largest_components is not None:
        floor_mesh = _keep_largest_components(floor_mesh, keep_largest_components)
        wall_mesh = _keep_largest_components(wall_mesh, keep_largest_components)

    logger.info(
        "segment_floor_walls: %s | floor_z=%.3f floor_faces=%d wall_faces=%d",
        mesh_path,
        floor_z,
        int(floor_mask.sum()),
        int(wall_mask.sum()),
    )

    return {
        "mesh": mesh,
        "floor_z": float(floor_z),
        "floor_face_mask": floor_mask,
        "wall_face_mask": wall_mask,
        "floor_mesh": floor_mesh,
        "wall_mesh": wall_mesh,
    }


def visualize_cameras(
    camera_poses,
    export_path: str = "camera_scene.obj",
    camera_scale: float = 1.0,
):
    """
    Visualize a list of camera poses and export to an OBJ or PLY file.

    Args:
        camera_poses: list of [4x4] camera pose matrices.
        export_path: output file path (OBJ or PLY).
    """
    scene = trimesh.Scene()

    # Loop through each camera pose
    for i, pose in enumerate(camera_poses):
        # Create a random color for each camera
        color = np.random.rand(3)

        # Create the camera mesh with the pose applied
        transformed_camera = create_transformed_camera(
            pose, scale=camera_scale, length=0.1, line_radius=0.005, color=color
        )
        scene.add_geometry(transformed_camera)

    # Export the scene to the specified file
    scene.export(export_path)
