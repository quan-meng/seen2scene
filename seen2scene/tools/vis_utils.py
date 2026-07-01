import matplotlib

matplotlib.use("Agg")  # Use headless backend before importing pyplot
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import torchvision
from typing import List, Optional, Dict, Union
from pathlib import Path
import json
import torch
import torch.nn.functional as F
import cv2
import io
from collections import OrderedDict
import plotly.graph_objects as go
import csv

_semantic_palette_cache = None
_clip_model_cache = None


def _embeddings_to_colors(embeddings, pca_components, embed_min, embed_max, mean):
    """Project CLIP embeddings to RGBA colors via PCA.

    Args:
        embeddings: [N, 512] or [1, 512] float64 CLIP embeddings.
        pca_components: [3, 512] PCA projection matrix.
        embed_min: [3] min values for normalization.
        embed_max: [3] max values for normalization.
        mean: [1, 512] embedding mean for centering.

    Returns:
        np.ndarray of shape [N, 4] (RGBA uint8).
    """
    projected = (embeddings - mean) @ pca_components.T  # [N, 3]
    rgb = np.clip(
        (projected - embed_min) / (embed_max - embed_min + 1e-8) * 255, 0, 255
    ).astype(np.uint8)
    return np.concatenate([rgb, np.full((len(rgb), 1), 255, dtype=np.uint8)], axis=1)


def _ensure_clip_cache():
    """Ensure CLIP embedding cache exists and return its contents.

    On first call, computes CLIP text embeddings for all semantic class names,
    fits PCA via SVD, and caches everything to clip_palette.npz.

    Returns:
        embeddings: [N, 512] float64, CLIP text embeddings for each class.
        pca_components: [3, 512], PCA projection matrix.
        embed_min: [3], min values for normalization.
        embed_max: [3], max values for normalization.
        mean: [1, 512], embedding mean.
        name_to_id: dict mapping class name -> category ID.
    """
    from seen2scene import ASSETS_DIR as assets_dir
    csv_path = assets_dir / "semantic_classes.csv"
    cache_path = assets_dir / "clip_palette.npz"

    # Read class names and IDs from CSV
    name_to_id = {}
    names = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name_to_id[row["name"]] = int(row["id"])
            names.append(row["name"])

    # Load from cache if available
    if cache_path.exists():
        data = np.load(cache_path)
        return (
            data["embeddings"], data["pca_components"],
            data["embed_min"], data["embed_max"], data["mean"], name_to_id,
        )

    # Compute CLIP text embeddings (first run only)
    from transformers import CLIPTextModel, AutoTokenizer

    model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32", use_safetensors=True)
    tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()

    encoding = tokenizer(
        names, max_length=77, truncation=True, padding="max_length", return_tensors="pt"
    )
    with torch.no_grad():
        text_features = model(input_ids=encoding["input_ids"]).pooler_output  # [N, 512]
    embeddings = text_features.cpu().numpy().astype(np.float64)

    # PCA via SVD
    mean = embeddings.mean(axis=0, keepdims=True)
    centered = embeddings - mean
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    pca_components = Vt[:3]  # [3, 512]
    projected = centered @ pca_components.T  # [N, 3]
    embed_min = projected.min(axis=0)
    embed_max = projected.max(axis=0)

    # Cache embeddings + PCA info
    np.savez(
        cache_path,
        embeddings=embeddings,
        pca_components=pca_components,
        embed_min=embed_min,
        embed_max=embed_max,
        mean=mean,
    )

    return embeddings, pca_components, embed_min, embed_max, mean, name_to_id


def get_clip_color(name: str) -> np.ndarray:
    """Get CLIP-based RGBA color for any object name.

    For known classes, uses cached embeddings. For unknown names, loads
    the CLIP model to compute the embedding on the fly.

    Args:
        name: Object name string.

    Returns:
        np.ndarray of shape [4] (RGBA uint8).
    """
    global _clip_model_cache
    embeddings, pca_components, embed_min, embed_max, mean, name_to_id = _ensure_clip_cache()

    # Known class — use cached embedding
    if name in name_to_id:
        idx = name_to_id[name]
        return _embeddings_to_colors(
            embeddings[idx : idx + 1], pca_components, embed_min, embed_max, mean
        )[0]

    # Unknown name — compute embedding with CLIP model
    if _clip_model_cache is None:
        from transformers import CLIPTextModel, AutoTokenizer

        model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32", use_safetensors=True)
        tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        model.eval()
        _clip_model_cache = (model, tokenizer)
    model, tokenizer = _clip_model_cache

    encoding = tokenizer(
        [name], max_length=77, truncation=True, padding="max_length", return_tensors="pt"
    )
    with torch.no_grad():
        text_features = model(input_ids=encoding["input_ids"]).pooler_output
    embedding = text_features.cpu().numpy().astype(np.float64)

    return _embeddings_to_colors(embedding, pca_components, embed_min, embed_max, mean)[0]


def load_semantic_palette():
    """Load CLIP-based semantic color palette.

    Colors are derived from CLIP text embeddings projected to RGB via PCA.

    Returns:
        palette: np.ndarray of shape [N, 4] (RGBA uint8)
        name_to_id: dict mapping category name -> category ID
    """
    global _semantic_palette_cache
    if _semantic_palette_cache is not None:
        return _semantic_palette_cache

    embeddings, pca_components, embed_min, embed_max, mean, name_to_id = _ensure_clip_cache()
    palette = _embeddings_to_colors(embeddings, pca_components, embed_min, embed_max, mean)
    _semantic_palette_cache = (palette, name_to_id)
    return palette, name_to_id


def fig_to_image_tensor(
    fig,
    *,
    resolution: Optional[int] = None,
    pad_inches: float = 0.0,
    dpi: Optional[int] = None,
    bbox_inches: str = "tight",
    format: str = "png",
) -> torch.Tensor:
    """Render a matplotlib figure to an RGB torch tensor [3, H, W] in [0, 1]."""
    buf = io.BytesIO()
    save_kwargs = {
        "format": format,
        "bbox_inches": bbox_inches,
        "pad_inches": pad_inches,
    }
    if dpi is not None:
        save_kwargs["dpi"] = dpi
    fig.savefig(buf, **save_kwargs)
    buf.seek(0)
    fig_array = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    fig_image = cv2.imdecode(fig_array, cv2.IMREAD_COLOR)
    buf.close()

    # Convert BGR to RGB and resize if needed
    fig_image = cv2.cvtColor(fig_image, cv2.COLOR_BGR2RGB)
    if resolution is not None and fig_image.shape[:2] != (resolution, resolution):
        fig_image = cv2.resize(fig_image, (resolution, resolution))

    return torch.from_numpy(fig_image).permute(2, 0, 1).float() / 255.0


def plotly_fig_to_image_tensor(
    fig,
    *,
    width: int = 1600,
    height: int = 800,
    scale: Optional[int] = None,
    format: str = "png",
) -> torch.Tensor:
    """Render a Plotly figure to an RGB torch tensor [3, H, W] in [0, 1].

    Args:
        fig: Plotly figure object
        width: Image width in pixels
        height: Image height in pixels
        scale: Scale factor for higher resolution (e.g., 2 for retina displays)
        format: Output format ("png", "jpeg", or "svg")

    Returns:
        Tensor of shape [3, H, W] with values in [0, 1]
    """
    # Export Plotly figure to image bytes
    export_kwargs = {"format": format, "width": width, "height": height}
    if scale is not None:
        export_kwargs["scale"] = scale

    img_bytes = fig.to_image(**export_kwargs)

    # Decode image bytes to numpy array
    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    # Convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Convert to torch tensor [3, H, W] in [0, 1]
    img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

    return img_tensor


def plot_object_bboxes(
    object_bboxes: List[np.ndarray],
    object_categories: List[np.ndarray],
    object_names: List[List[str]],
    patch_size: np.ndarray,
    palette: np.ndarray,
    resolution: int = 150,
    rotation: float = 0.0,
) -> torch.Tensor:
    """
    Plot object bounding boxes from top view (XY plane projection).

    Args:
        object_bboxes: List of numpy arrays, each of shape [N, 2, 3]
        object_categories: List of numpy arrays, each of shape [N] with category indices
        patch_size: Patch size as [6]
        object_names: List of lists of strings, object names for each batch item
        palette: Color palette array of shape [num_categories, 3] with RGB values 0-255
        resolution: Image resolution (height and width)
        rotation: Rotation angle in degrees around Z-axis for the view

    Returns:
        images: Tensor of shape [B, 3, H, W] where B is batch size
    """
    # Validate shape using first non-empty bbox array
    assert object_bboxes[0][0].shape == (
        2,
        3,
    ), f"object_bboxes must have shape [2, 3], got {object_bboxes[0][0].shape}"

    all_images = []
    for bboxes_i, categories_i, patch_size_i, names_i in zip(
        object_bboxes, object_categories, patch_size, object_names
    ):
        # Create figure for this view
        fig, ax = plt.subplots(figsize=(resolution / 100, resolution / 100), dpi=100)
        ax.set_xlim([0.0, patch_size_i[0]])
        ax.set_ylim([0.0, patch_size_i[1]])
        ax.set_aspect("equal")
        ax.axis("off")

        if len(bboxes_i) == 0:
            # Convert empty figure to image and continue
            img_tensor = fig_to_image_tensor(
                fig, resolution=resolution, pad_inches=0.0, dpi=100
            )
            plt.close(fig)
            all_images.append(img_tensor)
            continue

        sort_indices = np.argsort(bboxes_i[:, 1, 2])
        bboxes_i = bboxes_i[sort_indices, :, :2]  # [N, 2, 2]
        categories_i = categories_i[sort_indices]
        names_i = [names_i[idx] for idx in sort_indices]

        bboxes_i = np.clip(bboxes_i, a_min=0.0, a_max=patch_size_i[None, None, :2])

        if rotation == 0.0:
            bboxes_i = np.stack(
                [bboxes_i[..., 1], patch_size_i[0] - bboxes_i[..., 0]], axis=-1
            )
        elif rotation == 90.0:  # (x, y) -> (-x, -y)
            bboxes_i = patch_size_i[None, None, :2] - bboxes_i
        elif rotation == 180.0:  # (x, y) -> (-y, x)
            bboxes_i = np.stack(
                [patch_size_i[1] - bboxes_i[..., 1], bboxes_i[..., 0]], axis=-1
            )

        # Draw each bbox
        for cat_idx, bbox_xy, name in zip(categories_i, bboxes_i, names_i):
            color = palette[cat_idx] / 255.0  # Normalize to [0, 1]
            (xmin, ymin), (xmax, ymax) = bbox_xy
            corners = np.array([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]])

            rect = Polygon(
                corners, linewidth=0.0, edgecolor="none", facecolor=color, alpha=0.7
            )
            ax.add_patch(rect)

            # Two diagonal dotted lines connecting corners
            kwargs = {
                "linestyle": ":",
                "color": "black",
                "linewidth": 1.0,
                "alpha": 0.8,
            }
            ax.plot(
                [corners[0, 0], corners[2, 0]], [corners[0, 1], corners[2, 1]], **kwargs
            )
            ax.plot(
                [corners[1, 0], corners[3, 0]], [corners[1, 1], corners[3, 1]], **kwargs
            )
            cx, cy = corners.mean(axis=0)
            ax.text(
                cx,
                cy,
                name,
                ha="center",
                va="center",
                fontsize=8,
                color="black",
                bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=1.0),
            )

        # Convert figure to image
        img_tensor = fig_to_image_tensor(
            fig, resolution=resolution, pad_inches=0.0, dpi=100
        )
        plt.close(fig)
        all_images.append(img_tensor)

    # Stack images: Shape [B, 3, H, W]
    images = torch.stack(all_images)  # [B, 3, H, W]
    return images


def plot_layout(
    layout: Union[str, Path, Dict],
    output_path: Optional[Union[str, Path]] = None,
    figsize: tuple = (16, 7),
    dpi: int = 150
) -> plt.Figure:
    """
    Plot a room layout with 3D view and top-down floor plan.

    Args:
        layout: Either a path to a JSON file or a dict containing:
            - 'object_names': List of object name strings
            - 'object_bboxes': List of bounding boxes [x_min, y_min, z_min, x_max, y_max, z_max]
              where X, Y are floor plane coordinates and Z is height
        output_path: Optional path to save the figure. If None, figure is not saved.
        figsize: Figure size as (width, height) in inches.
        dpi: Resolution for saved figure.

    Returns:
        fig: The matplotlib figure object.
    """
    # Load layout data
    if isinstance(layout, (str, Path)):
        with open(layout, "r") as f:
            data = json.load(f)
    else:
        data = layout

    object_names = data["object_names"]
    object_bboxes = data["object_bboxes"]

    # Color palette for different objects
    colors = plt.cm.tab20(np.linspace(0, 1, len(object_names)))

    def get_box_vertices(bbox):
        """Get 8 vertices of a 3D box from [x_min, y_min, z_min, x_max, y_max, z_max]"""
        x_min, y_min, z_min, x_max, y_max, z_max = bbox
        vertices = [
            [x_min, y_min, z_min],
            [x_max, y_min, z_min],
            [x_max, y_max, z_min],
            [x_min, y_max, z_min],
            [x_min, y_min, z_max],
            [x_max, y_min, z_max],
            [x_max, y_max, z_max],
            [x_min, y_max, z_max],
        ]
        return np.array(vertices)

    def get_box_faces(vertices):
        """Get 6 faces of the box"""
        faces = [
            [vertices[0], vertices[1], vertices[2], vertices[3]],  # bottom
            [vertices[4], vertices[5], vertices[6], vertices[7]],  # top
            [vertices[0], vertices[1], vertices[5], vertices[4]],  # front
            [vertices[2], vertices[3], vertices[7], vertices[6]],  # back
            [vertices[0], vertices[3], vertices[7], vertices[4]],  # left
            [vertices[1], vertices[2], vertices[6], vertices[5]],  # right
        ]
        return faces

    # Compute axis limits from data
    all_coords = np.array(object_bboxes)
    x_max_lim = max(all_coords[:, 3].max(), all_coords[:, 0].max()) + 0.5
    y_max_lim = max(all_coords[:, 4].max(), all_coords[:, 1].max()) + 0.5
    z_max_lim = max(all_coords[:, 5].max(), all_coords[:, 2].max()) + 0.5

    # Create figure with 2 subplots: 3D view and top-down view
    fig = plt.figure(figsize=figsize)

    # 3D view - Z is height (vertical)
    ax1 = fig.add_subplot(121, projection="3d")
    for i, (name, bbox) in enumerate(zip(object_names, object_bboxes)):
        vertices = get_box_vertices(bbox)
        faces = get_box_faces(vertices)
        collection = Poly3DCollection(
            faces, alpha=0.6, facecolor=colors[i], edgecolor="black", linewidth=0.5
        )
        ax1.add_collection3d(collection)

        # Add label at center
        center = [
            (bbox[0] + bbox[3]) / 2,
            (bbox[1] + bbox[4]) / 2,
            (bbox[2] + bbox[5]) / 2,
        ]
        ax1.text(center[0], center[1], center[2], name, fontsize=7, ha="center")

    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z (height)")
    ax1.set_title("3D Layout View")
    ax1.set_xlim(0, x_max_lim)
    ax1.set_ylim(0, y_max_lim)
    ax1.set_zlim(0, z_max_lim)

    # Top-down view (X-Y plane, looking from above)
    ax2 = fig.add_subplot(122)
    for i, (name, bbox) in enumerate(zip(object_names, object_bboxes)):
        x_min, y_min, z_min, x_max, y_max, z_max = bbox
        width = x_max - x_min
        height = y_max - y_min
        rect = plt.Rectangle(
            (x_min, y_min),
            width,
            height,
            facecolor=colors[i],
            edgecolor="black",
            alpha=0.6,
            linewidth=1,
        )
        ax2.add_patch(rect)

        # Add label at center
        ax2.text(
            (x_min + x_max) / 2,
            (y_min + y_max) / 2,
            name,
            fontsize=7,
            ha="center",
            va="center",
        )

    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_title("Top-Down View (Floor Plan, X-Y plane)")
    ax2.set_xlim(-0.5, x_max_lim)
    ax2.set_ylim(-0.5, y_max_lim)
    ax2.set_aspect("equal")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=dpi, bbox_inches="tight")

    return fig


def batch2row_types2column(images_dict: OrderedDict[str, np.ndarray]) -> torch.Tensor:
    """
    Arrange samples as rows and image types as columns: each sample (batch item) becomes a row with its image types (src, gen, tgt) as columns.
    Args:
        images_dict: Dictionary of images. [B, V, 3, H, W], V == 1
    Returns:
        images: Tensor of images [B, T, 3, H, W] where each row contains a sample with src, gen, tgt concatenated
        titles: List of titles. [T]
    """
    assert all(
        img.shape[1] == 1 for img in images_dict.values()
    ), "Expected num_views=1 for all image types"
    titles = list(images_dict.keys())
    images_list = list(images_dict.values())
    images = np.concatenate(images_list, axis=1)  # [B, T, 3, H, W]
    images = torch.from_numpy(images)

    return images, titles


def views2row_types2column(images_dict: OrderedDict[str, torch.Tensor]) -> torch.Tensor:
    """
    Arrange views as rows and image types as columns: each view becomes a row with its image types (src, gen, tgt) as columns.
    Args:
        images_src: Dictionary of images. [B, V, 3, H, W]
    Returns:
        images: Tensor of images [V, T, 3, H, W] * B
        titles: List of titles. [T]
    """
    titles = list(images_dict.keys())
    images_list = list(images_dict.values())  # [B, V, 3, H, W]
    images = np.stack(images_list)  # [T, B, V, 3, H, W]
    images = images.transpose(1, 2, 0, 3, 4, 5)  # [B, V, T, 3, H, W]
    images = torch.from_numpy(images)

    return images, titles


def plot_titles(
    image: torch.Tensor,
    border_colors: torch.Tensor,
    titles: List[str],
    font_size: int = 10,
) -> torch.Tensor:
    # Convert torch tensor to numpy and transpose to correct format (H,W,3)
    img_np = image.permute(1, 2, 0).numpy()

    # Ensure img_np is in [0, 1] range
    img_np = np.clip(img_np, 0, 1)

    # Create figure with the image
    fig, ax = plt.subplots(
        figsize=(img_np.shape[1] / 100, img_np.shape[0] / 100), dpi=100
    )
    ax.imshow(img_np)
    ax.axis("off")

    # Create legend handles
    legend_handles = []
    for title, color in zip(titles, border_colors):
        color_rgb = (float(color[0]), float(color[1]), float(color[2]))
        handle = plt.Line2D(
            [0],
            [0],
            marker="s",
            color="w",
            markerfacecolor=color_rgb,
            markersize=font_size,
            label=title,
        )
        legend_handles.append(handle)

    # Add legend below the image
    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0),
        ncol=len(titles),
        frameon=False,
        fontsize=font_size,
    )

    # Adjust layout to fit legend
    plt.tight_layout()

    # Convert figure to image
    fig_tensor = fig_to_image_tensor(fig, pad_inches=0.02)
    plt.close(fig)

    return fig_tensor


def make_image_grid(
    images: torch.Tensor,
    titles: Optional[List[str]] = None,
    border_width: int = 3,
    nrow: int = 2,
    normalize: bool = False,
    scale_each: bool = False,
    pad_value: float = 1.0,
    padding: int = 3,
    font_size: int = 10,
) -> torch.Tensor:
    R, C, _, H, W = images.shape

    # Get colors from matplotlib colormap
    cmap = plt.get_cmap("tab10")
    border_colors = torch.tensor([cmap(i)[:3] for i in np.linspace(0, 1, C)])  # [T, 3]

    # Reshape images and add padding
    images = images.reshape(-1, 3, H, W)  # [R*T, 3, H, W]
    bordered_images = F.pad(images, (border_width,) * 4, mode="constant", value=0)

    # Repeat border colors for each batch, [R*T, 3, 1, 1]
    border_colors_ = border_colors.repeat(R, 1, 1).reshape(-1, 3, 1, 1)

    # Apply borders efficiently using broadcasting
    bordered_images[:, :, :border_width, :] = border_colors_  # top
    bordered_images[:, :, -border_width:, :] = border_colors_  # bottom
    bordered_images[:, :, :, :border_width] = border_colors_  # left
    bordered_images[:, :, :, -border_width:] = border_colors_  # right

    # Create grids
    image_grid = []
    for i in range(R):
        batch_images = bordered_images[i * C : (i + 1) * C]
        batch_grid = torchvision.utils.make_grid(
            batch_images,
            nrow=C,
            normalize=normalize,
            padding=padding,
            scale_each=scale_each,
            pad_value=pad_value,
        )
        image_grid.append(batch_grid)

    image_grid = torchvision.utils.make_grid(
        image_grid,
        nrow=min(nrow, len(image_grid)),
        normalize=normalize,
        scale_each=scale_each,
        padding=padding,
        pad_value=pad_value,
    )

    return plot_titles(image_grid, border_colors, titles, font_size=font_size)


def plot_attention_bias(
    voxel_xyz: np.ndarray,
    object_bboxes: np.ndarray,
    attn_bias: np.ndarray,
    object_names: Optional[List[str]] = None,
    output_path: Optional[Union[str, Path]] = None,
    figsize: tuple = (18, 10),
    dpi: int = 150,
    max_voxels: int = 5000,
    cmap: str = "RdBu_r",
) -> plt.Figure:
    """
    Visualize attention bias between voxels and object bounding boxes.

    Creates a multi-panel figure:
    - Top row: 3D scatter plots showing voxels colored by attention bias to each object
    - Bottom left: Heatmap of full attention bias matrix [voxels x objects]
    - Bottom right: Distribution of attention bias values per object

    Args:
        voxel_xyz: Voxel positions [N, 3] in world coordinates
        object_bboxes: Object bounding boxes [M, 2, 3] where [:, 0, :] is min and [:, 1, :] is max
        attn_bias: Attention bias matrix [N, M] (positive = inside/attend more, negative = outside)
        object_names: Optional list of object names [M]
        output_path: Optional path to save the figure
        figsize: Figure size as (width, height)
        dpi: Resolution for saved figure
        max_voxels: Maximum number of voxels to plot (randomly sampled if exceeded)
        cmap: Colormap for attention bias visualization

    Returns:
        fig: The matplotlib figure object
    """
    N, M = attn_bias.shape

    # Convert torch tensors to numpy if needed
    if isinstance(voxel_xyz, torch.Tensor):
        voxel_xyz = voxel_xyz.detach().cpu().numpy()
    if isinstance(object_bboxes, torch.Tensor):
        object_bboxes = object_bboxes.detach().cpu().numpy()
    if isinstance(attn_bias, torch.Tensor):
        attn_bias = attn_bias.detach().cpu().numpy()

    # Default object names
    if object_names is None:
        object_names = [f"Object {i}" for i in range(M)]

    # Subsample voxels if too many
    if N > max_voxels:
        indices = np.random.choice(N, max_voxels, replace=False)
        indices = np.sort(indices)
        voxel_xyz_plot = voxel_xyz[indices]
        attn_bias_plot = attn_bias[indices]
    else:
        voxel_xyz_plot = voxel_xyz
        attn_bias_plot = attn_bias

    # Determine grid layout
    n_cols = min(4, M)
    n_rows_3d = (M + n_cols - 1) // n_cols

    # Create figure with gridspec for flexible layout
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(
        n_rows_3d + 1,
        n_cols + 1,
        height_ratios=[1] * n_rows_3d + [1.2],
        width_ratios=[1] * n_cols + [0.3],
        hspace=0.3,
        wspace=0.3,
    )

    # Compute display values: binary inside/outside with smooth transition
    # Map: inside (>0) -> [0.5, 1.0], outside (<0) -> [0.0, 0.5]
    # This ensures clear red/blue distinction while showing distance gradient
    from matplotlib.colors import TwoSlopeNorm

    # Normalize distances for display
    pos_max = (
        max(attn_bias_plot[attn_bias_plot >= 0].max(), 0.001)
        if (attn_bias_plot >= 0).any()
        else 0.001
    )
    neg_min = (
        min(attn_bias_plot[attn_bias_plot < 0].min(), -0.001)
        if (attn_bias_plot < 0).any()
        else -0.001
    )

    # Create display values that emphasize inside/outside distinction
    # Inside (>= 0): scale to [0.5, 1.5] so always appears red
    # Outside (< 0): scale to [-1.5, -0.5] so always appears blue
    attn_bias_plot_display = np.where(
        attn_bias_plot >= 0,  # >= 0 so boundary is red
        0.5 + (attn_bias_plot / pos_max),  # inside: 0.5 to 1.5
        -0.5 + (attn_bias_plot / abs(neg_min)),  # outside: -1.5 to -0.5
    )

    norm = TwoSlopeNorm(vmin=-1.5, vcenter=0, vmax=1.5)

    def draw_bbox_wireframe(ax, bbox, color="black", alpha=0.8, linewidth=1.5):
        """Draw a 3D bounding box as wireframe."""
        min_pt, max_pt = bbox[0], bbox[1]

        # 8 vertices of the box
        vertices = np.array(
            [
                [min_pt[0], min_pt[1], min_pt[2]],
                [max_pt[0], min_pt[1], min_pt[2]],
                [max_pt[0], max_pt[1], min_pt[2]],
                [min_pt[0], max_pt[1], min_pt[2]],
                [min_pt[0], min_pt[1], max_pt[2]],
                [max_pt[0], min_pt[1], max_pt[2]],
                [max_pt[0], max_pt[1], max_pt[2]],
                [min_pt[0], max_pt[1], max_pt[2]],
            ]
        )

        # 12 edges connecting the vertices
        edges = [
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 0],  # bottom face
            [4, 5],
            [5, 6],
            [6, 7],
            [7, 4],  # top face
            [0, 4],
            [1, 5],
            [2, 6],
            [3, 7],  # vertical edges
        ]

        for edge in edges:
            points = vertices[edge]
            ax.plot3D(*points.T, color=color, alpha=alpha, linewidth=linewidth)

    # Top rows: 3D scatter plots for each object
    for obj_idx in range(M):
        row = obj_idx // n_cols
        col = obj_idx % n_cols

        ax = fig.add_subplot(gs[row, col], projection="3d")

        # Scatter voxels colored by display values (ensures clear inside/outside colors)
        # Sort by bias so positive (inside) voxels are drawn on top
        bias_to_obj_display = attn_bias_plot_display[:, obj_idx]
        bias_to_obj_raw = attn_bias_plot[:, obj_idx]
        sort_idx = np.argsort(
            bias_to_obj_raw
        )  # ascending: negative first, positive last

        # Use larger markers for inside voxels (>= 0 includes boundary)
        sizes = np.where(bias_to_obj_raw >= 0, 15, 3)  # inside: 15, outside: 3

        scatter = ax.scatter(
            voxel_xyz_plot[sort_idx, 0],
            voxel_xyz_plot[sort_idx, 1],
            voxel_xyz_plot[sort_idx, 2],
            c=bias_to_obj_display[sort_idx],
            cmap=cmap,
            norm=norm,
            s=sizes[sort_idx],
            alpha=0.7,
        )

        # Draw all bboxes as wireframes (current one highlighted)
        for i, bbox in enumerate(object_bboxes):
            color = "red" if i == obj_idx else "gray"
            alpha = 1.0 if i == obj_idx else 0.3
            lw = 2.0 if i == obj_idx else 0.5
            draw_bbox_wireframe(ax, bbox, color=color, alpha=alpha, linewidth=lw)

        ax.set_xlabel("X", fontsize=8)
        ax.set_ylabel("Y", fontsize=8)
        ax.set_zlabel("Z", fontsize=8)
        ax.set_title(f"{object_names[obj_idx]}", fontsize=10, fontweight="bold")
        ax.tick_params(labelsize=6)

    # Add colorbar for 3D plots
    cbar_ax = fig.add_subplot(gs[:n_rows_3d, -1])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Attention Bias\n(red=inside, blue=outside)", fontsize=10)

    # Bottom left: Heatmap of attention bias matrix
    ax_heatmap = fig.add_subplot(gs[-1, : n_cols // 2])

    # Downsample for heatmap if too many voxels
    # Apply same display transform for heatmap
    pos_max_full = (
        max(attn_bias[attn_bias >= 0].max(), 0.001) if (attn_bias >= 0).any() else 0.001
    )
    neg_min_full = (
        min(attn_bias[attn_bias < 0].min(), -0.001) if (attn_bias < 0).any() else -0.001
    )
    attn_bias_display_full = np.where(
        attn_bias >= 0,  # >= 0 so boundary is red
        0.5 + (attn_bias / pos_max_full),
        -0.5 + (attn_bias / abs(neg_min_full)),
    )

    heatmap_max_voxels = 500
    if N > heatmap_max_voxels:
        step = N // heatmap_max_voxels
        attn_bias_heatmap = attn_bias_display_full[::step]
        ylabel = f"Voxels (subsampled 1:{step})"
    else:
        attn_bias_heatmap = attn_bias_display_full
        ylabel = "Voxels"

    im = ax_heatmap.imshow(
        attn_bias_heatmap, aspect="auto", cmap=cmap, norm=norm, interpolation="nearest"
    )
    ax_heatmap.set_xlabel("Objects", fontsize=10)
    ax_heatmap.set_ylabel(ylabel, fontsize=10)
    ax_heatmap.set_title("Attention Bias Matrix", fontsize=11, fontweight="bold")
    ax_heatmap.set_xticks(range(M))
    ax_heatmap.set_xticklabels(object_names, rotation=45, ha="right", fontsize=8)
    fig.colorbar(im, ax=ax_heatmap, shrink=0.8)

    # Bottom right: Distribution of attention bias per object
    ax_dist = fig.add_subplot(gs[-1, n_cols // 2 : -1])

    positions = np.arange(M)
    bp = ax_dist.boxplot(
        [attn_bias[:, i] for i in range(M)],
        positions=positions,
        widths=0.6,
        patch_artist=True,
    )

    colors_box = plt.cm.tab10(np.linspace(0, 1, M))
    for patch, color in zip(bp["boxes"], colors_box):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax_dist.axhline(y=0, color="black", linestyle="--", alpha=0.5, linewidth=1)
    ax_dist.set_xticks(positions)
    ax_dist.set_xticklabels(object_names, rotation=45, ha="right", fontsize=8)
    ax_dist.set_xlabel("Objects", fontsize=10)
    ax_dist.set_ylabel("Attention Bias", fontsize=10)
    ax_dist.set_title("Bias Distribution per Object", fontsize=11, fontweight="bold")
    ax_dist.grid(axis="y", alpha=0.3)

    # Add summary statistics
    n_inside = (attn_bias >= 0).sum(axis=0)
    n_total = attn_bias.shape[0]
    summary_text = "Inside ratio: " + ", ".join(
        [f"{object_names[i]}: {n_inside[i]/n_total:.1%}" for i in range(M)]
    )
    fig.text(0.5, 0.02, summary_text, ha="center", fontsize=9, style="italic")

    plt.suptitle(
        "Cross-Attention Bias: Voxel-to-Object Spatial Prior",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved attention bias visualization to {output_path}")

    return fig


def plot_attention_bias_single(
    voxel_xyz: np.ndarray,
    bbox: np.ndarray,
    attn_bias: np.ndarray,
    object_name: str = "Object",
    output_path: Optional[Union[str, Path]] = None,
    figsize: tuple = (14, 5),
    dpi: int = 150,
    max_voxels: int = 10000,
    cmap: str = "RdBu_r",
) -> plt.Figure:
    """
    Visualize attention bias for a single object with multiple views.

    Creates a figure with:
    - Left: 3D scatter plot of voxels colored by attention bias
    - Middle: Top-down (XY) view
    - Right: Side (XZ) view

    Args:
        voxel_xyz: Voxel positions [N, 3]
        bbox: Single bounding box [2, 3] where [0] is min and [1] is max
        attn_bias: Attention bias [N] for this object
        object_name: Name of the object
        output_path: Optional path to save the figure
        figsize: Figure size
        dpi: Resolution
        max_voxels: Maximum voxels to plot
        cmap: Colormap

    Returns:
        fig: The matplotlib figure object
    """
    N = len(attn_bias)

    # Convert torch tensors to numpy
    if isinstance(voxel_xyz, torch.Tensor):
        voxel_xyz = voxel_xyz.detach().cpu().numpy()
    if isinstance(bbox, torch.Tensor):
        bbox = bbox.detach().cpu().numpy()
    if isinstance(attn_bias, torch.Tensor):
        attn_bias = attn_bias.detach().cpu().numpy()

    # Subsample if needed
    if N > max_voxels:
        indices = np.random.choice(N, max_voxels, replace=False)
        voxel_xyz = voxel_xyz[indices]
        attn_bias = attn_bias[indices]

    # Create display values that emphasize inside/outside distinction
    from matplotlib.colors import TwoSlopeNorm

    pos_max = (
        max(attn_bias[attn_bias >= 0].max(), 0.001) if (attn_bias >= 0).any() else 0.001
    )
    neg_min = (
        min(attn_bias[attn_bias < 0].min(), -0.001) if (attn_bias < 0).any() else -0.001
    )

    attn_bias_display = np.where(
        attn_bias >= 0,  # >= 0 so boundary voxels are also red
        0.5 + (attn_bias / pos_max),  # inside: 0.5 to 1.5
        -0.5 + (attn_bias / abs(neg_min)),  # outside: -1.5 to -0.5
    )

    norm = TwoSlopeNorm(vmin=-1.5, vcenter=0, vmax=1.5)

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Sort by bias so inside voxels (positive) are drawn on top
    sort_idx = np.argsort(attn_bias)
    voxel_xyz_sorted = voxel_xyz[sort_idx]
    attn_bias_sorted = attn_bias_display[sort_idx]
    sizes = np.where(
        attn_bias[sort_idx] >= 0, 15, 3
    )  # larger for inside (>= 0 includes boundary)

    # 3D view
    ax3d = fig.add_subplot(131, projection="3d")
    scatter = ax3d.scatter(
        voxel_xyz_sorted[:, 0],
        voxel_xyz_sorted[:, 1],
        voxel_xyz_sorted[:, 2],
        c=attn_bias_sorted,
        cmap=cmap,
        norm=norm,
        s=sizes,
        alpha=0.7,
    )

    # Draw bbox wireframe
    min_pt, max_pt = bbox[0], bbox[1]
    for i in range(2):
        for j in range(2):
            ax3d.plot3D(
                [min_pt[0], max_pt[0]],
                [min_pt[1] if j == 0 else max_pt[1]] * 2,
                [min_pt[2] if i == 0 else max_pt[2]] * 2,
                "r-",
                linewidth=2,
            )
            ax3d.plot3D(
                [min_pt[0] if j == 0 else max_pt[0]] * 2,
                [min_pt[1], max_pt[1]],
                [min_pt[2] if i == 0 else max_pt[2]] * 2,
                "r-",
                linewidth=2,
            )
            ax3d.plot3D(
                [min_pt[0] if j == 0 else max_pt[0]] * 2,
                [min_pt[1] if i == 0 else max_pt[1]] * 2,
                [min_pt[2], max_pt[2]],
                "r-",
                linewidth=2,
            )

    ax3d.set_xlabel("X")
    ax3d.set_ylabel("Y")
    ax3d.set_zlabel("Z")
    ax3d.set_title(f"3D View: {object_name}")
    fig.colorbar(scatter, ax=ax3d, shrink=0.6, label="Attention Bias")

    # Top-down view (XY) - only show voxels within bbox Z range
    ax_xy = axes[1]
    # Filter to voxels within the Z slice of the bbox (with small margin)
    z_margin = (max_pt[2] - min_pt[2]) * 0.1
    z_mask = (voxel_xyz[:, 2] >= min_pt[2] - z_margin) & (
        voxel_xyz[:, 2] <= max_pt[2] + z_margin
    )
    voxel_xy_slice = voxel_xyz[z_mask]
    bias_xy_slice = attn_bias[z_mask]
    bias_xy_display = (
        attn_bias_display[z_mask]
        if len(attn_bias_display) == len(attn_bias)
        else attn_bias_display[z_mask[sort_idx]]
    )
    # Recompute display for slice
    pos_max_xy = (
        max(bias_xy_slice[bias_xy_slice >= 0].max(), 0.001)
        if (bias_xy_slice >= 0).any()
        else 0.001
    )
    neg_min_xy = (
        min(bias_xy_slice[bias_xy_slice < 0].min(), -0.001)
        if (bias_xy_slice < 0).any()
        else -0.001
    )
    bias_xy_display = np.where(
        bias_xy_slice >= 0,  # >= 0 for boundary
        0.5 + (bias_xy_slice / pos_max_xy),
        -0.5 + (bias_xy_slice / abs(neg_min_xy)),
    )
    sort_xy = np.argsort(bias_xy_slice)
    sizes_xy = np.where(bias_xy_slice[sort_xy] >= 0, 15, 3)

    scatter_xy = ax_xy.scatter(
        voxel_xy_slice[sort_xy, 0],
        voxel_xy_slice[sort_xy, 1],
        c=bias_xy_display[sort_xy],
        cmap=cmap,
        norm=norm,
        s=sizes_xy,
        alpha=0.7,
    )
    rect = plt.Rectangle(
        (min_pt[0], min_pt[1]),
        max_pt[0] - min_pt[0],
        max_pt[1] - min_pt[1],
        fill=False,
        edgecolor="red",
        linewidth=2,
    )
    ax_xy.add_patch(rect)
    ax_xy.set_xlabel("X")
    ax_xy.set_ylabel("Y")
    ax_xy.set_title(f"Top-Down View (XY)\nZ slice: [{min_pt[2]:.1f}, {max_pt[2]:.1f}]")
    ax_xy.set_aspect("equal")
    fig.colorbar(scatter_xy, ax=ax_xy, shrink=0.6)

    # Side view (XZ) - only show voxels within bbox Y range
    ax_xz = axes[2]
    y_margin = (max_pt[1] - min_pt[1]) * 0.1
    y_mask = (voxel_xyz[:, 1] >= min_pt[1] - y_margin) & (
        voxel_xyz[:, 1] <= max_pt[1] + y_margin
    )
    voxel_xz_slice = voxel_xyz[y_mask]
    bias_xz_slice = attn_bias[y_mask]
    # Recompute display for slice
    pos_max_xz = (
        max(bias_xz_slice[bias_xz_slice >= 0].max(), 0.001)
        if (bias_xz_slice >= 0).any()
        else 0.001
    )
    neg_min_xz = (
        min(bias_xz_slice[bias_xz_slice < 0].min(), -0.001)
        if (bias_xz_slice < 0).any()
        else -0.001
    )
    bias_xz_display = np.where(
        bias_xz_slice >= 0,  # >= 0 for boundary
        0.5 + (bias_xz_slice / pos_max_xz),
        -0.5 + (bias_xz_slice / abs(neg_min_xz)),
    )
    sort_xz = np.argsort(bias_xz_slice)
    sizes_xz = np.where(bias_xz_slice[sort_xz] >= 0, 15, 3)

    scatter_xz = ax_xz.scatter(
        voxel_xz_slice[sort_xz, 0],
        voxel_xz_slice[sort_xz, 2],
        c=bias_xz_display[sort_xz],
        cmap=cmap,
        norm=norm,
        s=sizes_xz,
        alpha=0.7,
    )
    rect = plt.Rectangle(
        (min_pt[0], min_pt[2]),
        max_pt[0] - min_pt[0],
        max_pt[2] - min_pt[2],
        fill=False,
        edgecolor="red",
        linewidth=2,
    )
    ax_xz.add_patch(rect)
    ax_xz.set_xlabel("X")
    ax_xz.set_ylabel("Z")
    ax_xz.set_title(f"Side View (XZ)\nY slice: [{min_pt[1]:.1f}, {max_pt[1]:.1f}]")
    ax_xz.set_aspect("equal")
    fig.colorbar(scatter_xz, ax=ax_xz, shrink=0.6)

    # Statistics
    n_inside = (attn_bias >= 0).sum()
    pct_inside = n_inside / len(attn_bias) * 100
    fig.suptitle(
        f"{object_name}: {pct_inside:.1f}% voxels inside (bias >= 0)",
        fontsize=12,
        fontweight="bold",
    )

    plt.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=dpi, bbox_inches="tight")

    return fig
