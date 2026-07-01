#!/usr/bin/env python3
"""
Hierarchical 3D scene layout generation: House → Rooms → Furniture → Items

Usage:
    python models/layout/get_layout_hierarchical.py --num 1 --house_size 8.0 8.0 2.8 --num_rooms 3
    python models/layout/get_layout_hierarchical.py --num 5 --house_size 10.0 10.0 2.8 --num_rooms 4 --model gpt-4o

Logic:
    1. Generate [type_1, type_2, ..., type_l] rooms for a House
    2. For each type_i room: Generate furniture objects [fur_1, ..., fur_m]
    3. For each furniture object fur_i: Generate items [item_1, ..., item_n] on it

Coordinate System:
    - Each object uses its parent's coordinate system as origin
    - Room coordinates are relative to house origin
    - Furniture coordinates are relative to room origin
    - Item coordinates are relative to furniture origin
"""

import csv
import os
import sys
import json
import re
import subprocess
import tempfile
import hashlib
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, List, Optional, Dict, Any
import tyro

from seen2scene.configs.dataset import Dataset, LLMsLayout
from seen2scene.configs.opt import LOG_ROOT
from seen2scene.models.layout.hierarchical_utils import hierarchical_to_meta
from seen2scene.tools.vis_utils import plot_layout
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def get_layout_categories(
    csv_path: str = None,
    exclude_set: Optional[set[str]] = None,
) -> List[str]:
    """Get categories suitable for layout generation."""
    if csv_path is None:
        from seen2scene import ASSETS_DIR
        csv_path = str(ASSETS_DIR / "semantic_classes.csv")
    classes = {}
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            classes[row["name"]] = int(row["id"])
    return [name for name in classes.keys() if name not in exclude_set]


valid_categories = get_layout_categories(exclude_set=set(Dataset.category_exclude))

# Note: We no longer hardcode room types or furniture types.
# When use_semantic_mapping=True: LLM generates from valid_categories
# When use_semantic_mapping=False: LLM generates arbitrary object names


@dataclass
class HierarchicalConfig:
    """Configuration for hierarchical layout generation."""

    num: int = 1
    """Number of house layouts to generate."""

    house_size: Tuple[float, float, float] = (4.5, 4.5, 2.8)
    """Size of the house in meters (width, depth, height)."""

    num_rooms: int = 1
    """Number of rooms in the house."""

    out_dir: Optional[str] = None
    """Output directory for generated layouts. If None, uses LLMsLayout.root_dir from configs."""

    scene_type: Optional[str] = None
    """Scene type subdirectory (deprecated - no longer used in dataset format)."""

    model: str = ""
    """Model to use for generation (e.g., o4-mini, gpt-4o)."""

    prefix: str = "house"
    """Prefix for output filenames (not used in dataset mode, kept for compatibility)."""

    start_index: int = 0
    """Starting index for output filenames (not used in dataset mode, kept for compatibility)."""

    verbose: bool = False
    """Print verbose output."""

    use_semantic_mapping: bool = True
    """If True, constrain object names to predefined semantic categories from CSV.
    If False, allow LLM to generate arbitrary object names freely."""

    use_dataset_format: bool = True
    """If True, save in dataset-compatible format (scene_hash/meta.json).
    If False, use legacy flat file format."""

    save_visualization: bool = True
    """If True, save visualization images to scene directory."""

    prompt: Optional[str] = None
    """Optional text prompt to guide layout generation (e.g., 'a dining room for 10-20 people')."""

    prompt_file: Optional[str] = None
    """Path to a .txt file with one prompt per line. Each line generates a separate scene.
    When set, overrides --prompt and --num (one scene per line)."""


def generate_house_prompt(config: HierarchicalConfig, prompt: Optional[str] = None) -> str:
    """Generate prompt for house-level layout (room placement)."""

    if config.use_semantic_mapping:
        room_constraint = "Common room types include: bedroom, living_room, kitchen, bathroom, dining_room, office, etc."
    else:
        room_constraint = "Use natural room type names (e.g., 'master bedroom', 'open kitchen', 'study room', 'gym', etc.)."

    llm_prompt = f"""You are a house layout generator. You generate realistic room arrangements for houses.

You work in a 3D coordinate system where (0, 0, 0) is the house origin at the bottom-left corner.
- X axis: width (horizontal)
- Y axis: depth (horizontal)
- Z axis: height (vertical)

TASK: Generate a house layout with {config.num_rooms} rooms.

House dimensions: {config.house_size[0]}m × {config.house_size[1]}m × {config.house_size[2]}m

Room naming: {room_constraint}

OUTPUT FORMAT (JSON only):
{{
  "rooms": [
    {{
      "type": "room_type",
      "position": [x, y, z],
      "size": [width, depth, height]
    }}
  ]
}}

RULES:
1. Room positions are relative to house origin (0, 0, 0)
2. Rooms must fit within house bounds: 0 ≤ x ≤ {config.house_size[0]}, 0 ≤ y ≤ {config.house_size[1]}
3. Rooms should not overlap significantly (small shared walls are OK)
4. All rooms should have z=0 and height={config.house_size[2]}m
5. Room sizes should be realistic (bedrooms: 3-5m, living rooms: 4-7m, bathrooms: 2-3m, etc.)
6. Create a diverse mix of room types suitable for a house

Generate exactly {config.num_rooms} rooms. Only output JSON, nothing else.
"""
    if prompt:
        llm_prompt += f"""
USER REQUEST (your layout MUST satisfy this):
{prompt}
"""
    return llm_prompt


def generate_room_furniture_prompt(
    room_type: str,
    room_size: Tuple[float, float, float],
    use_semantic_mapping: bool,
    prompt: Optional[str] = None,
) -> str:
    """Generate prompt for room-level layout (furniture placement)."""

    if use_semantic_mapping:
        category_constraint = f"Object names must be from these categories: {', '.join(valid_categories)}"
    else:
        category_constraint = "Use natural, descriptive object names (e.g., 'king size bed', 'leather sofa', 'floor lamp', 'coffee maker', etc.)"

    llm_prompt = f"""You are a room layout generator. You generate realistic furniture arrangements for rooms.

You work in a 3D coordinate system where (0, 0, 0) is the room origin at the bottom-left corner.
- X axis: width (horizontal)
- Y axis: depth (horizontal)
- Z axis: height (vertical)

TASK: Generate furniture layout for a {room_type}.

Room dimensions: {room_size[0]}m × {room_size[1]}m × {room_size[2]}m

Object naming: {category_constraint}

OUTPUT FORMAT (JSON only):
{{
  "furniture": [
    {{
      "name": "object_name",
      "bbox": [[x_min, y_min, z_min], [x_max, y_max, z_max]]
    }}
  ]
}}

RULES:
1. Furniture positions are relative to room origin (0, 0, 0)
2. All furniture must fit within room bounds
3. Furniture should not overlap
4. Place furniture realistically for a {room_type}:
   - Beds/sofas against walls
   - Tables in center or near walls
   - Lamps on tables or floor
   - Carpets on floor (z=0)
5. Use realistic furniture sizes in meters
6. Leave clear walking space (0.8-1m pathways)

Generate 3-8 furniture pieces. Only output JSON, nothing else.
"""
    if prompt:
        llm_prompt += f"""
USER REQUEST (furniture MUST satisfy this):
{prompt}

IMPORTANT: Generate as many individual furniture pieces as needed to satisfy the request above. Do not limit yourself to a small number.
Only output JSON, nothing else.
"""
    return llm_prompt


def generate_furniture_items_prompt(
    furniture_name: str,
    furniture_bbox: List[List[float]],
    room_type: str,
    use_semantic_mapping: bool,
    prompt: Optional[str] = None,
) -> str:
    """Generate prompt for furniture-level layout (items on furniture).

    The LLM decides whether items are appropriate for this furniture.
    """

    furniture_size = [
        furniture_bbox[1][0] - furniture_bbox[0][0],
        furniture_bbox[1][1] - furniture_bbox[0][1],
        furniture_bbox[1][2] - furniture_bbox[0][2]
    ]
    surface_height = furniture_bbox[1][2]  # Top surface Z coordinate

    if use_semantic_mapping:
        item_constraint = f"Item names must be from these categories: {', '.join(valid_categories)}"
    else:
        item_constraint = "Use natural, specific item names (e.g., 'wireless mouse', 'coffee mug', 'reading glasses', 'phone charger', etc.)"

    llm_prompt = f"""You are an item placement generator. Your job is to decide if items should be placed on furniture, and if so, generate them.

You work in a 3D coordinate system where (0, 0, 0) is the furniture origin at the bottom-left corner.
- X axis: width
- Y axis: depth
- Z axis: height

CONTEXT:
- Room type: {room_type}
- Furniture: {furniture_name}
- Furniture dimensions: {furniture_size[0]:.2f}m × {furniture_size[1]:.2f}m × {furniture_size[2]:.2f}m
- Surface height (Z): {surface_height:.2f}m (relative to room)

TASK:
1. First, decide if this furniture typically has items placed on it
   - Tables, desks, nightstands, shelves, counters → YES
   - Chairs, sofas, beds, trash bins, lamps → NO

2. If YES, generate realistic items ON TOP of the furniture
3. If NO, return empty items list

Item naming: {item_constraint}

OUTPUT FORMAT (JSON only):
{{
  "has_items": true/false,
  "items": [
    {{
      "name": "item_name",
      "bbox": [[x_min, y_min, z_min], [x_max, y_max, z_max]]
    }}
  ]
}}

RULES:
1. Item positions are relative to furniture origin (0, 0, 0)
2. Items must be ON the surface: z_min should be at or near surface height ({surface_height:.2f}m)
3. Items must fit within furniture top surface: 0 ≤ x ≤ {furniture_size[0]:.2f}, 0 ≤ y ≤ {furniture_size[1]:.2f}
4. Items should not overlap with each other
5. Use realistic item sizes (typically 0.05-0.3m)
6. Place items naturally based on furniture type and room context
7. Generate a reasonable number of items (can be empty if not appropriate)

EXAMPLES:
- Chair: {{"has_items": false, "items": []}}
- Desk: {{"has_items": true, "items": [{{"name": "lamp", "bbox": [[0.1, 0.1, {surface_height:.2f}], [0.3, 0.3, {surface_height+0.3:.2f}]]}}]}}

Only output JSON, nothing else.
"""
    if prompt:
        llm_prompt += f"""
SCENE CONTEXT (items should be coherent with this):
{prompt}
"""
    return llm_prompt


def extract_json(text: str) -> Optional[dict]:
    """Extract JSON from text that may contain markdown code blocks."""
    # Try to find JSON in code blocks
    json_pattern = r"```(?:json)?\s*\n?([\s\S]*?)\n?```"
    matches = re.findall(json_pattern, text)

    for match in matches:
        try:
            return json.loads(match.strip())
        except json.JSONDecodeError:
            continue

    # Try to parse entire text as JSON
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    # Try to find JSON object pattern
    json_obj_pattern = r'\{[\s\S]*\}'
    matches = re.findall(json_obj_pattern, text)

    for match in matches:
        try:
            return json.loads(match)
        except json.JSONDecodeError:
            continue

    return None


def call_codex(prompt: str, model: str = "", verbose: bool = False, max_retries: int = 3) -> Optional[str]:
    """Call Codex CLI and return response text."""

    for attempt in range(max_retries):
        try:
            if verbose:
                print(f"    Codex attempt {attempt + 1}...", end=" ", flush=True)

            # Create temp file for output
            with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
                output_file = f.name

            cmd = [
                "codex", "exec",
                "--skip-git-repo-check",
                "-o", output_file,
                "--json",
                "-",
            ]
            if model:
                cmd.insert(2, "-m")
                cmd.insert(3, model)

            result = subprocess.run(
                cmd,
                input=prompt,
                capture_output=True,
                text=True,
                timeout=120,
            )

            # Parse JSONL output
            response_text = ""
            for line in result.stdout.strip().split("\n"):
                if not line:
                    continue
                try:
                    event = json.loads(line)
                    if event.get("type") == "item.completed":
                        item = event.get("item", {})
                        if item.get("type") == "agent_message":
                            response_text += item.get("text", "")
                    elif event.get("type") == "message" and event.get("role") == "assistant":
                        content = event.get("content", [])
                        for c in content:
                            if c.get("type") in ("output_text", "text"):
                                response_text += c.get("text", "")
                except json.JSONDecodeError:
                    continue

            # Check output file
            if os.path.exists(output_file):
                with open(output_file, "r") as f:
                    file_content = f.read().strip()
                    if file_content and not response_text:
                        response_text = file_content
                os.unlink(output_file)

            if response_text:
                if verbose:
                    print(f"OK ({len(response_text)} chars)")
                return response_text

            if verbose:
                print("Empty response")

        except subprocess.TimeoutExpired:
            if verbose:
                print("Timeout")
        except Exception as e:
            if verbose:
                print(f"Error: {e}")

    return None


def generate_hierarchical_layout(config: HierarchicalConfig) -> Optional[Dict[str, Any]]:
    """Generate a complete hierarchical house layout."""

    # Step 1: Generate house-level layout (rooms)
    if config.verbose:
        print("  [1/3] Generating house layout (rooms)...")

    house_prompt = generate_house_prompt(config, prompt=config.prompt)
    house_response = call_codex(house_prompt, config.model, config.verbose)

    if not house_response:
        if config.verbose:
            print("    FAILED to generate house layout")
        return None

    house_data = extract_json(house_response)
    if not house_data or "rooms" not in house_data:
        if config.verbose:
            print("    FAILED to parse house layout")
        return None

    rooms = house_data["rooms"]
    if config.verbose:
        print(f"    Generated {len(rooms)} rooms")

    # Step 2: Generate furniture for each room
    if config.verbose:
        print(f"  [2/3] Generating furniture for {len(rooms)} rooms...")

    for i, room in enumerate(rooms):
        room_type = room.get("type", "room")
        room_size = tuple(room.get("size", [3, 3, 2.8]))

        if config.verbose:
            print(f"    Room {i+1}/{len(rooms)} ({room_type})...", end=" ", flush=True)

        furniture_prompt = generate_room_furniture_prompt(
            room_type, room_size, config.use_semantic_mapping,
            prompt=config.prompt,
        )
        furniture_response = call_codex(furniture_prompt, config.model, verbose=False)

        if not furniture_response:
            if config.verbose:
                print("FAILED")
            room["furniture"] = []
            continue

        furniture_data = extract_json(furniture_response)
        if not furniture_data or "furniture" not in furniture_data:
            if config.verbose:
                print("FAILED (parse)")
            room["furniture"] = []
            continue

        room["furniture"] = furniture_data["furniture"]
        if config.verbose:
            print(f"OK ({len(room['furniture'])} furniture)")

    # Step 3: Generate items for each furniture piece
    if config.verbose:
        print(f"  [3/3] Generating items on furniture...")

    total_furniture = sum(len(room.get("furniture", [])) for room in rooms)
    furniture_count = 0

    for room_idx, room in enumerate(rooms):
        room_type = room.get("type", "room")

        for fur_idx, furniture in enumerate(room.get("furniture", [])):
            furniture_count += 1

            furniture_name = furniture.get("name", "furniture")
            furniture_bbox = furniture.get("bbox", [[0, 0, 0], [1, 1, 1]])

            if config.verbose:
                print(f"    Furniture {furniture_count}/{total_furniture} ({furniture_name})...", end=" ", flush=True)

            # Ask LLM if this furniture should have items
            items_prompt = generate_furniture_items_prompt(
                furniture_name,
                furniture_bbox,
                room_type,
                config.use_semantic_mapping,
                prompt=config.prompt,
            )

            items_response = call_codex(items_prompt, config.model, verbose=False)

            if not items_response:
                furniture["items"] = []
                if config.verbose:
                    print("FAILED")
                continue

            items_data = extract_json(items_response)
            if not items_data:
                furniture["items"] = []
                if config.verbose:
                    print("FAILED (parse)")
                continue

            # Handle response with has_items field
            has_items = items_data.get("has_items", True)  # Default to True for backwards compatibility
            items_list = items_data.get("items", [])

            if not has_items:
                # LLM decided no items should be on this furniture
                furniture["items"] = []
                if config.verbose:
                    print("No items (LLM decision)")
            else:
                furniture["items"] = items_list
                if config.verbose:
                    print(f"OK ({len(furniture['items'])} items)")

    # Build final hierarchical structure
    layout = {
        "house": {
            "size": list(config.house_size),
            "rooms": rooms
        }
    }
    if config.prompt:
        layout["prompt"] = config.prompt

    return layout


def flatten_to_world_coords(layout: Dict[str, Any]) -> Dict[str, Any]:
    """Convert hierarchical layout to flat world coordinates for compatibility."""

    object_names = []
    object_bboxes = []

    house = layout["house"]

    for room in house.get("rooms", []):
        room_pos = room.get("position", [0, 0, 0])

        for furniture in room.get("furniture", []):
            # Add furniture in world coordinates
            fur_name = furniture.get("name", "object")
            fur_bbox = furniture.get("bbox", [[0, 0, 0], [1, 1, 1]])

            # Convert to world coords
            world_bbox = [
                [room_pos[0] + fur_bbox[0][0], room_pos[1] + fur_bbox[0][1], room_pos[2] + fur_bbox[0][2]],
                [room_pos[0] + fur_bbox[1][0], room_pos[1] + fur_bbox[1][1], room_pos[2] + fur_bbox[1][2]]
            ]

            object_names.append(fur_name)
            object_bboxes.append([
                world_bbox[0][0], world_bbox[0][1], world_bbox[0][2],
                world_bbox[1][0], world_bbox[1][1], world_bbox[1][2]
            ])

            # Add items in world coordinates
            for item in furniture.get("items", []):
                item_name = item.get("name", "item")
                item_bbox = item.get("bbox", [[0, 0, 0], [0.1, 0.1, 0.1]])

                # Convert to world coords (room + furniture position)
                world_item_bbox = [
                    [room_pos[0] + fur_bbox[0][0] + item_bbox[0][0],
                     room_pos[1] + fur_bbox[0][1] + item_bbox[0][1],
                     room_pos[2] + fur_bbox[0][2] + item_bbox[0][2]],
                    [room_pos[0] + fur_bbox[0][0] + item_bbox[1][0],
                     room_pos[1] + fur_bbox[0][1] + item_bbox[1][1],
                     room_pos[2] + fur_bbox[0][2] + item_bbox[1][2]]
                ]

                object_names.append(item_name)
                object_bboxes.append([
                    world_item_bbox[0][0], world_item_bbox[0][1], world_item_bbox[0][2],
                    world_item_bbox[1][0], world_item_bbox[1][1], world_item_bbox[1][2]
                ])

    return {
        "object_names": object_names,
        "object_bboxes": object_bboxes
    }


def visualize_layout_with_rooms(
    layout: Dict[str, Any],
    meta_data: Dict[str, Any],
    output_path: str,
    figsize: tuple = (16, 7),
    dpi: int = 150,
) -> None:
    """Generate visualization with room information overlaid.

    Args:
        layout: Hierarchical layout structure
        meta_data: Meta format (flat objects)
        output_path: Path to save visualization
        figsize: Figure size
        dpi: Resolution
    """
    # Convert meta format to flat format for base visualization
    flat_for_viz = {
        "object_names": meta_data["object_names"],
        "object_bboxes": [
            [bbox[0][0], bbox[0][1], bbox[0][2], bbox[1][0], bbox[1][1], bbox[1][2]]
            for bbox in meta_data["object_bboxes"]
        ]
    }

    # Create base visualization
    fig = plot_layout(flat_for_viz, output_path=None, figsize=figsize, dpi=dpi)

    # Get the two axes (3D view and top view)
    axes = fig.get_axes()
    if len(axes) >= 2:
        ax_3d = axes[0]  # 3D subplot
        ax_top = axes[1]  # Top-down subplot

        house = layout["house"]
        house_size = house["size"]

        # Add room boundaries and labels to 3D view
        for room in house["rooms"]:
            room_pos = room["position"]
            room_size = room["size"]
            room_type = room["type"]

            # Draw room floor boundary in 3D
            corners = [
                [room_pos[0], room_pos[1], room_pos[2]],
                [room_pos[0] + room_size[0], room_pos[1], room_pos[2]],
                [room_pos[0] + room_size[0], room_pos[1] + room_size[1], room_pos[2]],
                [room_pos[0], room_pos[1] + room_size[1], room_pos[2]],
            ]

            for i in range(4):
                j = (i + 1) % 4
                ax_3d.plot(
                    [corners[i][0], corners[j][0]],
                    [corners[i][1], corners[j][1]],
                    [corners[i][2], corners[j][2]],
                    color="blue", linewidth=2, linestyle="--", alpha=0.7
                )

            # Add room label in 3D
            center = [
                room_pos[0] + room_size[0] / 2,
                room_pos[1] + room_size[1] / 2,
                room_pos[2] + room_size[2] + 0.2
            ]
            ax_3d.text(
                center[0], center[1], center[2],
                room_type, fontsize=10, ha="center", weight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue",
                         alpha=0.8, edgecolor="blue")
            )

        # Add room boundaries and labels to top view
        for room in house["rooms"]:
            room_pos = room["position"]
            room_size = room["size"]
            room_type = room["type"]

            # Draw room boundary rectangle
            room_rect = Rectangle(
                (room_pos[0], room_pos[1]),
                room_size[0], room_size[1],
                fill=False,
                edgecolor="blue",
                linewidth=2,
                linestyle="--",
                alpha=0.7
            )
            ax_top.add_patch(room_rect)

            # Add room label
            center = (room_pos[0] + room_size[0] / 2, room_pos[1] + room_size[1] / 2)
            label_y = room_pos[1] + room_size[1] - 0.3
            ax_top.text(
                center[0], label_y, room_type,
                fontsize=9, ha="center", va="top", weight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue",
                         alpha=0.8, edgecolor="blue")
            )

    # Save the figure
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def generate_scene_type(layout: Dict[str, Any]) -> str:
    """Generate scene type - defaults to 'house' for residential layouts."""
    return "house"


def generate_scene_hash(layout: Dict[str, Any]) -> str:
    """Generate a unique hash for a scene based on its content."""
    # Use JSON representation to create a consistent hash
    scene_str = json.dumps(layout, sort_keys=True)
    hash_obj = hashlib.sha256(scene_str.encode())
    return hash_obj.hexdigest()[:16]  # Use first 16 characters


def check_codex_login() -> bool:
    """Check if codex is logged in."""
    try:
        result = subprocess.run(
            ["codex", "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.returncode == 0
    except Exception:
        return False


def main(config: HierarchicalConfig):
    """Main entry point for hierarchical layout generation."""

    # Check for codex
    if not check_codex_login():
        print("Error: Codex CLI not found or not working")
        print("Install: npm install -g @openai/codex")
        return

    # Determine output directory
    if config.out_dir is None:
        llms_layout = LLMsLayout()
        out_dir = Path(llms_layout.root_dir)
    else:
        out_dir = Path(config.out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    # If prompt_file is provided, read prompts from file (one per line)
    if config.prompt_file is not None:
        prompt_file_path = Path(config.prompt_file)
        if not prompt_file_path.exists():
            print(f"Error: Prompt file not found: {config.prompt_file}")
            return
        with open(prompt_file_path, "r") as f:
            prompts = [line.strip() for line in f if line.strip()]
        if not prompts:
            print(f"Error: Prompt file is empty: {config.prompt_file}")
            return
        config.num = len(prompts)
        print(f"Loaded {len(prompts)} prompts from {config.prompt_file}")
    else:
        prompts = None

    print(f"Generating {config.num} hierarchical house layouts...")
    print(f"House size: {config.house_size[0]}×{config.house_size[1]}×{config.house_size[2]}m")
    print(f"Rooms per house: {config.num_rooms}")
    print(f"Output directory: {out_dir}")
    if config.prompt:
        print(f"Prompt: {config.prompt}")
    print(f"Format: {'Dataset-compatible' if config.use_dataset_format else 'Legacy flat files'}\n")

    success_count = 0

    for i in range(config.num):
        idx = config.start_index + i

        # Override prompt from file if available
        if prompts is not None:
            config.prompt = prompts[i]

        print(f"[{i+1}/{config.num}] Generating house {idx:04d}...")
        if config.prompt:
            print(f"  Prompt: {config.prompt}")

        layout = generate_hierarchical_layout(config)

        if layout is None:
            print(f"  FAILED to generate complete layout\n")
            continue

        # Count objects
        num_rooms = len(layout["house"]["rooms"])
        num_furniture = sum(len(r.get("furniture", [])) for r in layout["house"]["rooms"])
        num_items = sum(
            len(f.get("items", []))
            for r in layout["house"]["rooms"]
            for f in r.get("furniture", [])
        )

        if config.use_dataset_format:
            # Dataset-compatible format: scene_hash/meta.json

            # Generate scene hash
            scene_hash = generate_scene_hash(layout)

            # Create scene directory directly under root
            scene_dir = out_dir / scene_hash
            scene_dir.mkdir(parents=True, exist_ok=True)

            # Save hierarchical layout (optional, for debugging/visualization)
            hierarchical_path = scene_dir / "layout_hierarchical.json"
            with open(hierarchical_path, "w") as f:
                json.dump(layout, f, indent=2)

            # Save meta.json in dataset format
            meta_data = hierarchical_to_meta(layout)
            if config.prompt:
                meta_data["prompt"] = config.prompt
            meta_path = scene_dir / "meta.json"
            with open(meta_path, "w") as f:
                json.dump(meta_data, f, indent=2)

            # Generate structure mesh and full scene mesh
            try:
                from seen2scene.models.layout.hierarchical_utils import generate_structure_mesh
                from seen2scene.models.layout.build_scene_mesh import build_scene_mesh
                structure_mesh = generate_structure_mesh(layout)
                structure_path = scene_dir / "structure.ply"
                structure_mesh.export(str(structure_path))

                scene_mesh = build_scene_mesh(layout)
                scene_mesh_path = scene_dir / "scene.ply"
                scene_mesh.export(str(scene_mesh_path))
                if config.verbose:
                    print(f"  Meshes saved: {structure_path.name}, {scene_mesh_path.name}")
            except Exception as e:
                if config.verbose:
                    print(f"  Warning: Failed to generate meshes: {e}")

            # Generate and save visualization with room information
            if config.save_visualization:
                try:
                    viz_path = scene_dir / "layout_visualization.png"
                    visualize_layout_with_rooms(layout, meta_data, str(viz_path), figsize=(16, 7), dpi=150)
                    if config.verbose:
                        print(f"  Visualization saved to {viz_path.name}")
                except Exception as e:
                    if config.verbose:
                        print(f"  Warning: Failed to generate visualization: {e}")

            print(f"  SUCCESS: {num_rooms} rooms, {num_furniture} furniture, {num_items} items")
            print(f"  Scene: {scene_hash}")
            print(f"  Saved: {meta_path.relative_to(out_dir)}\n")

        else:
            # Legacy format: flat files with index
            hierarchical_path = out_dir / f"{config.prefix}_hierarchical_{idx:04d}.json"
            with open(hierarchical_path, "w") as f:
                json.dump(layout, f, indent=2)

            # Save flattened layout for compatibility
            flat_layout = flatten_to_world_coords(layout)
            flat_path = out_dir / f"{config.prefix}_flat_{idx:04d}.json"
            with open(flat_path, "w") as f:
                json.dump(flat_layout, f, indent=2)

            # Generate structure mesh and full scene mesh
            try:
                from seen2scene.models.layout.hierarchical_utils import generate_structure_mesh
                from seen2scene.models.layout.build_scene_mesh import build_scene_mesh
                structure_mesh = generate_structure_mesh(layout)
                structure_path = out_dir / f"{config.prefix}_structure_{idx:04d}.ply"
                structure_mesh.export(str(structure_path))

                scene_mesh = build_scene_mesh(layout)
                scene_mesh_path = out_dir / f"{config.prefix}_scene_{idx:04d}.ply"
                scene_mesh.export(str(scene_mesh_path))
                if config.verbose:
                    print(f"  Meshes saved: {structure_path.name}, {scene_mesh_path.name}")
            except Exception as e:
                if config.verbose:
                    print(f"  Warning: Failed to generate meshes: {e}")

            print(f"  SUCCESS: {num_rooms} rooms, {num_furniture} furniture, {num_items} items")
            print(f"  Saved: {hierarchical_path.name}, {flat_path.name}\n")

        success_count += 1

    print(f"Generated {success_count}/{config.num} layouts successfully")


if __name__ == "__main__":
    tyro.cli(main)
