#!/usr/bin/env python3
"""
Utility functions for working with hierarchical layouts.

Features:
- Convert hierarchical → flat (world coordinates)
- Convert flat → hierarchical (with clustering)
- Validate hierarchical layouts
- Visualize layout statistics
- Print layout in tree format
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass


@dataclass
class LayoutStats:
    """Statistics for a hierarchical layout."""

    num_rooms: int
    num_furniture: int
    num_items: int
    furniture_with_items: int
    furniture_without_items: int
    avg_items_per_furniture: float
    rooms_by_type: Dict[str, int]


def hierarchical_to_flat(layout: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert hierarchical layout to flat world coordinates.

    Args:
        layout: Hierarchical layout with house → rooms → furniture → items

    Returns:
        Flat layout with object_names and object_bboxes in world coordinates
    """
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
                [
                    room_pos[0] + fur_bbox[0][0],
                    room_pos[1] + fur_bbox[0][1],
                    room_pos[2] + fur_bbox[0][2],
                ],
                [
                    room_pos[0] + fur_bbox[1][0],
                    room_pos[1] + fur_bbox[1][1],
                    room_pos[2] + fur_bbox[1][2],
                ],
            ]

            object_names.append(fur_name)
            object_bboxes.append(
                [
                    world_bbox[0][0],
                    world_bbox[0][1],
                    world_bbox[0][2],
                    world_bbox[1][0],
                    world_bbox[1][1],
                    world_bbox[1][2],
                ]
            )

            # Add items in world coordinates
            for item in furniture.get("items", []):
                item_name = item.get("name", "item")
                item_bbox = item.get("bbox", [[0, 0, 0], [0.1, 0.1, 0.1]])

                # Convert to world coords (room + furniture position)
                world_item_bbox = [
                    [
                        room_pos[0] + fur_bbox[0][0] + item_bbox[0][0],
                        room_pos[1] + fur_bbox[0][1] + item_bbox[0][1],
                        room_pos[2] + fur_bbox[0][2] + item_bbox[0][2],
                    ],
                    [
                        room_pos[0] + fur_bbox[0][0] + item_bbox[1][0],
                        room_pos[1] + fur_bbox[0][1] + item_bbox[1][1],
                        room_pos[2] + fur_bbox[0][2] + item_bbox[1][2],
                    ],
                ]

                object_names.append(item_name)
                object_bboxes.append(
                    [
                        world_item_bbox[0][0],
                        world_item_bbox[0][1],
                        world_item_bbox[0][2],
                        world_item_bbox[1][0],
                        world_item_bbox[1][1],
                        world_item_bbox[1][2],
                    ]
                )

    return {"object_names": object_names, "object_bboxes": object_bboxes}


def hierarchical_to_meta(layout: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert hierarchical layout to meta.json format compatible with dataset loader.

    This format matches the structure expected by dataset/common.py:load_scene_meta()

    Args:
        layout: Hierarchical layout with house → rooms → furniture → items

    Returns:
        Dictionary in meta.json format with:
        - scene_box: [[xmin, ymin, zmin], [xmax, ymax, zmax]] - computed from object bounds and house height
        - object_names: List of all object names (furniture + items)
        - object_bboxes: List of bounding boxes [[xmin, ymin, zmin], [xmax, ymax, zmax]]
    """
    import numpy as np

    house = layout["house"]
    house_size = house["size"]

    object_names = []
    object_bboxes = []

    for room in house.get("rooms", []):
        room_pos = room.get("position", [0, 0, 0])

        for furniture in room.get("furniture", []):
            fur_name = furniture.get("name", "object")
            fur_bbox = furniture.get("bbox", [[0, 0, 0], [1, 1, 1]])

            # Convert to world coordinates in [[min], [max]] format
            world_bbox = [
                [
                    room_pos[0] + fur_bbox[0][0],
                    room_pos[1] + fur_bbox[0][1],
                    room_pos[2] + fur_bbox[0][2],
                ],
                [
                    room_pos[0] + fur_bbox[1][0],
                    room_pos[1] + fur_bbox[1][1],
                    room_pos[2] + fur_bbox[1][2],
                ],
            ]

            object_names.append(fur_name)
            object_bboxes.append(world_bbox)

            # Add items in world coordinates
            for item in furniture.get("items", []):
                item_name = item.get("name", "item")
                item_bbox = item.get("bbox", [[0, 0, 0], [0.1, 0.1, 0.1]])

                # Convert to world coords (room + furniture position)
                world_item_bbox = [
                    [
                        room_pos[0] + fur_bbox[0][0] + item_bbox[0][0],
                        room_pos[1] + fur_bbox[0][1] + item_bbox[0][1],
                        room_pos[2] + fur_bbox[0][2] + item_bbox[0][2],
                    ],
                    [
                        room_pos[0] + fur_bbox[0][0] + item_bbox[1][0],
                        room_pos[1] + fur_bbox[0][1] + item_bbox[1][1],
                        room_pos[2] + fur_bbox[0][2] + item_bbox[1][2],
                    ],
                ]

                object_names.append(item_name)
                object_bboxes.append(world_item_bbox)

    # Compute scene_box from object bounding boxes
    if len(object_bboxes) > 0:
        all_bboxes = np.array(object_bboxes)  # [N, 2, 3]
        scene_max = all_bboxes[:, 1, :].max(axis=0).tolist()  # [3]
        # Use house height as the z_max (ceiling height)
        scene_max[2] = house_size[2]
    else:
        # Default to house bounds if no objects
        scene_max = house_size

    scene_box = [0.0, 0.0, 0.0, scene_max]

    return {
        "scene_box": scene_box,
        "object_names": object_names,
        "object_bboxes": object_bboxes,
    }


def validate_hierarchical_layout(layout: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate a hierarchical layout for consistency and correctness.

    Returns:
        (is_valid, error_messages)
    """
    errors = []

    # Check top-level structure
    if "house" not in layout:
        errors.append("Missing 'house' key")
        return False, errors

    house = layout["house"]

    if "size" not in house:
        errors.append("Missing 'house.size'")
    else:
        house_size = house["size"]
        if len(house_size) != 3:
            errors.append(f"house.size should be [x, y, z], got {house_size}")

    if "rooms" not in house:
        errors.append("Missing 'house.rooms'")
        return len(errors) == 0, errors

    rooms = house["rooms"]

    # Validate each room
    for i, room in enumerate(rooms):
        room_prefix = f"room[{i}]"

        # Check required fields
        if "type" not in room:
            errors.append(f"{room_prefix}: Missing 'type'")

        if "position" not in room:
            errors.append(f"{room_prefix}: Missing 'position'")
        elif len(room["position"]) != 3:
            errors.append(
                f"{room_prefix}.position: Expected [x, y, z], got {room['position']}"
            )

        if "size" not in room:
            errors.append(f"{room_prefix}: Missing 'size'")
        elif len(room["size"]) != 3:
            errors.append(f"{room_prefix}.size: Expected [x, y, z], got {room['size']}")

        # Check if room fits in house
        if "position" in room and "size" in room and "size" in house:
            room_pos = room["position"]
            room_size = room["size"]
            house_size = house["size"]

            room_end = [room_pos[i] + room_size[i] for i in range(3)]

            for axis, (end, limit) in enumerate(zip(room_end, house_size)):
                if end > limit:
                    errors.append(
                        f"{room_prefix}: Extends beyond house bounds on axis {axis} "
                        f"({end:.2f} > {limit:.2f})"
                    )

        # Validate furniture
        if "furniture" not in room:
            errors.append(f"{room_prefix}: Missing 'furniture'")
            continue

        for j, furniture in enumerate(room["furniture"]):
            fur_prefix = f"{room_prefix}.furniture[{j}]"

            if "name" not in furniture:
                errors.append(f"{fur_prefix}: Missing 'name'")

            if "bbox" not in furniture:
                errors.append(f"{fur_prefix}: Missing 'bbox'")
            else:
                bbox = furniture["bbox"]
                if len(bbox) != 2 or len(bbox[0]) != 3 or len(bbox[1]) != 3:
                    errors.append(
                        f"{fur_prefix}.bbox: Expected [[x,y,z], [x,y,z]], got {bbox}"
                    )
                else:
                    # Check if furniture fits in room
                    if "size" in room:
                        room_size = room["size"]
                        for axis in range(3):
                            if bbox[1][axis] > room_size[axis]:
                                errors.append(
                                    f"{fur_prefix}: Extends beyond room bounds on axis {axis} "
                                    f"({bbox[1][axis]:.2f} > {room_size[axis]:.2f})"
                                )

            # Validate items
            if "items" not in furniture:
                errors.append(
                    f"{fur_prefix}: Missing 'items' (use empty list if no items)"
                )
                continue

            for k, item in enumerate(furniture["items"]):
                item_prefix = f"{fur_prefix}.items[{k}]"

                if "name" not in item:
                    errors.append(f"{item_prefix}: Missing 'name'")

                if "bbox" not in item:
                    errors.append(f"{item_prefix}: Missing 'bbox'")
                else:
                    item_bbox = item["bbox"]
                    if (
                        len(item_bbox) != 2
                        or len(item_bbox[0]) != 3
                        or len(item_bbox[1]) != 3
                    ):
                        errors.append(
                            f"{item_prefix}.bbox: Expected [[x,y,z], [x,y,z]], got {item_bbox}"
                        )
                    else:
                        # Check if item fits on furniture surface
                        if "bbox" in furniture:
                            fur_bbox = furniture["bbox"]
                            fur_size = [
                                fur_bbox[1][i] - fur_bbox[0][i] for i in range(3)
                            ]

                            for axis in range(2):  # Only check X and Y (surface)
                                if item_bbox[1][axis] > fur_size[axis]:
                                    errors.append(
                                        f"{item_prefix}: Extends beyond furniture surface on axis {axis}"
                                    )

    return len(errors) == 0, errors


def compute_layout_stats(layout: Dict[str, Any]) -> LayoutStats:
    """Compute statistics for a hierarchical layout."""
    house = layout["house"]
    rooms = house.get("rooms", [])

    num_rooms = len(rooms)
    num_furniture = 0
    num_items = 0
    furniture_with_items = 0
    furniture_without_items = 0
    rooms_by_type = {}

    for room in rooms:
        room_type = room.get("type", "unknown")
        rooms_by_type[room_type] = rooms_by_type.get(room_type, 0) + 1

        for furniture in room.get("furniture", []):
            num_furniture += 1
            items = furniture.get("items", [])
            num_items += len(items)

            if len(items) > 0:
                furniture_with_items += 1
            else:
                furniture_without_items += 1

    avg_items = num_items / num_furniture if num_furniture > 0 else 0.0

    return LayoutStats(
        num_rooms=num_rooms,
        num_furniture=num_furniture,
        num_items=num_items,
        furniture_with_items=furniture_with_items,
        furniture_without_items=furniture_without_items,
        avg_items_per_furniture=avg_items,
        rooms_by_type=rooms_by_type,
    )


def print_layout_tree(layout: Dict[str, Any], indent: int = 0):
    """Print hierarchical layout in tree format."""
    house = layout["house"]
    house_size = house["size"]

    print(
        "  " * indent + f"🏠 House ({house_size[0]}×{house_size[1]}×{house_size[2]}m)"
    )

    for i, room in enumerate(house.get("rooms", [])):
        room_type = room.get("type", "room")
        room_pos = room.get("position", [0, 0, 0])
        room_size = room.get("size", [0, 0, 0])

        print(
            "  " * (indent + 1)
            + f"📦 Room {i+1}: {room_type} @ ({room_pos[0]:.1f}, {room_pos[1]:.1f}) "
            f"[{room_size[0]}×{room_size[1]}×{room_size[2]}m]"
        )

        for j, furniture in enumerate(room.get("furniture", [])):
            fur_name = furniture.get("name", "furniture")
            fur_bbox = furniture.get("bbox", [[0, 0, 0], [0, 0, 0]])
            fur_size = [fur_bbox[1][k] - fur_bbox[0][k] for k in range(3)]

            num_items = len(furniture.get("items", []))

            print(
                "  " * (indent + 2)
                + f"🪑 {fur_name} @ ({fur_bbox[0][0]:.1f}, {fur_bbox[0][1]:.1f}) "
                f"[{fur_size[0]:.2f}×{fur_size[1]:.2f}×{fur_size[2]:.2f}m] "
                f"({num_items} items)"
            )

            for k, item in enumerate(furniture.get("items", [])):
                item_name = item.get("name", "item")
                item_bbox = item.get("bbox", [[0, 0, 0], [0, 0, 0]])

                print(
                    "  " * (indent + 3)
                    + f"📌 {item_name} @ ({item_bbox[0][0]:.2f}, {item_bbox[0][1]:.2f}, {item_bbox[0][2]:.2f})"
                )


def print_layout_stats(layout: Dict[str, Any]):
    """Print detailed statistics for a hierarchical layout."""
    stats = compute_layout_stats(layout)

    print("=" * 60)
    print("Layout Statistics")
    print("=" * 60)
    print(f"Rooms:     {stats.num_rooms}")
    print(f"Furniture: {stats.num_furniture}")
    print(f"Items:     {stats.num_items}")
    print()
    print(
        f"Furniture with items:    {stats.furniture_with_items} ({stats.furniture_with_items/stats.num_furniture*100:.1f}%)"
    )
    print(
        f"Furniture without items: {stats.furniture_without_items} ({stats.furniture_without_items/stats.num_furniture*100:.1f}%)"
    )
    print(f"Avg items per furniture: {stats.avg_items_per_furniture:.2f}")
    print()
    print("Rooms by type:")
    for room_type, count in sorted(stats.rooms_by_type.items()):
        print(f"  {room_type}: {count}")
    print("=" * 60)


def _find_room_adjacencies(rooms: List[Dict[str, Any]], tolerance: float = 0.15) -> List[Dict[str, Any]]:
    """
    Find pairs of adjacent rooms by checking if their faces align.

    Returns list of dicts with keys:
        room_i, room_j, axis ('x' or 'y'), wall_pos, shared_min, shared_max
    """
    adjacencies = []

    for i in range(len(rooms)):
        pi = rooms[i].get("position", [0, 0, 0])
        si = rooms[i].get("size", [0, 0, 0])

        for j in range(i + 1, len(rooms)):
            pj = rooms[j].get("position", [0, 0, 0])
            sj = rooms[j].get("size", [0, 0, 0])

            # Check X-axis adjacency: i's east face vs j's west face (or vice versa)
            for (a, b, ai, bi) in [(i, j, pi, pj), (j, i, pj, pi)]:
                pa = [pi, pj][0] if a == i else [pi, pj][1]
                sa_ = [si, sj][0] if a == i else [si, sj][1]
                pb = [pi, pj][0] if b == i else [pi, pj][1]
                sb_ = [si, sj][0] if b == i else [si, sj][1]

                # a's east face (x = pa[0]+sa_[0]) ≈ b's west face (x = pb[0])
                if abs((pa[0] + sa_[0]) - pb[0]) < tolerance:
                    # Check Y overlap
                    a_y_min, a_y_max = pa[1], pa[1] + sa_[1]
                    b_y_min, b_y_max = pb[1], pb[1] + sb_[1]
                    shared_min = max(a_y_min, b_y_min)
                    shared_max = min(a_y_max, b_y_max)
                    if shared_max - shared_min > tolerance:
                        adjacencies.append({
                            "room_i": a, "room_j": b,
                            "axis": "x", "wall_pos": pb[0],
                            "shared_min": shared_min, "shared_max": shared_max,
                        })
                        break  # Only one adjacency per pair on this axis

                # a's north face (y = pa[1]+sa_[1]) ≈ b's south face (y = pb[1])
                if abs((pa[1] + sa_[1]) - pb[1]) < tolerance:
                    # Check X overlap
                    a_x_min, a_x_max = pa[0], pa[0] + sa_[0]
                    b_x_min, b_x_max = pb[0], pb[0] + sb_[0]
                    shared_min = max(a_x_min, b_x_min)
                    shared_max = min(a_x_max, b_x_max)
                    if shared_max - shared_min > tolerance:
                        adjacencies.append({
                            "room_i": a, "room_j": b,
                            "axis": "y", "wall_pos": pb[1],
                            "shared_min": shared_min, "shared_max": shared_max,
                        })
                        break

    return adjacencies


def _create_room_floor(pos, size, thickness=0.05):
    """Create a floor mesh for a room."""
    import trimesh
    floor = trimesh.creation.box(extents=[size[0], size[1], thickness])
    floor.apply_translation([
        pos[0] + size[0] / 2,
        pos[1] + size[1] / 2,
        -thickness / 2,
    ])
    return floor


def _wall_segment(center, extents):
    """Create a single wall box at given center with given extents."""
    import trimesh
    box = trimesh.creation.box(extents=extents)
    box.apply_translation(center)
    return box


def _create_room_walls(pos, size, wall_thickness=0.1, doors=None):
    """
    Create wall meshes for a room, with door openings cut out.

    Args:
        pos: [x, y, z] room position (world coords)
        size: [w, d, h] room size
        wall_thickness: thickness of walls
        doors: list of dicts with keys: wall ('south','north','east','west'),
               door_min, door_max (along wall's parallel axis), door_height
    """
    if doors is None:
        doors = []

    meshes = []
    x0, y0 = pos[0], pos[1]
    w, d, h = size[0], size[1], size[2]
    wt = wall_thickness

    # Group doors by wall
    wall_doors = {"south": [], "north": [], "west": [], "east": []}
    for door in doors:
        wall_doors[door["wall"]].append(door)

    # Sort doors on each wall by position
    for wall_name in wall_doors:
        wall_doors[wall_name].sort(key=lambda dd: dd["door_min"])

    # South wall (y = y0, along X)
    _build_wall_with_doors(
        meshes, wall_doors["south"],
        wall_start=x0, wall_end=x0 + w, wall_height=h,
        wall_center_perp=y0 + wt / 2, wall_thickness=wt,
        axis="x", perp_axis="y",
    )

    # North wall (y = y0 + d, along X)
    _build_wall_with_doors(
        meshes, wall_doors["north"],
        wall_start=x0, wall_end=x0 + w, wall_height=h,
        wall_center_perp=y0 + d - wt / 2, wall_thickness=wt,
        axis="x", perp_axis="y",
    )

    # West wall (x = x0, along Y)
    _build_wall_with_doors(
        meshes, wall_doors["west"],
        wall_start=y0, wall_end=y0 + d, wall_height=h,
        wall_center_perp=x0 + wt / 2, wall_thickness=wt,
        axis="y", perp_axis="x",
    )

    # East wall (x = x0 + w, along Y)
    _build_wall_with_doors(
        meshes, wall_doors["east"],
        wall_start=y0, wall_end=y0 + d, wall_height=h,
        wall_center_perp=x0 + w - wt / 2, wall_thickness=wt,
        axis="y", perp_axis="x",
    )

    return meshes


def _build_wall_with_doors(meshes, doors, wall_start, wall_end, wall_height,
                           wall_center_perp, wall_thickness, axis, perp_axis):
    """Build wall segments with door cutouts and door panels along a given axis."""
    if not doors:
        # Solid wall
        length = wall_end - wall_start
        center_along = (wall_start + wall_end) / 2
        if axis == "x":
            meshes.append(_wall_segment(
                [center_along, wall_center_perp, wall_height / 2],
                [length, wall_thickness, wall_height],
            ))
        else:
            meshes.append(_wall_segment(
                [wall_center_perp, center_along, wall_height / 2],
                [wall_thickness, length, wall_height],
            ))
        return

    # Build segments around doors
    current_start = wall_start
    for door in doors:
        d_min = door["door_min"]
        d_max = door["door_max"]
        d_h = door["door_height"]

        # Left segment (from current_start to door_min)
        if d_min - current_start > 0.001:
            seg_len = d_min - current_start
            center_along = (current_start + d_min) / 2
            if axis == "x":
                meshes.append(_wall_segment(
                    [center_along, wall_center_perp, wall_height / 2],
                    [seg_len, wall_thickness, wall_height],
                ))
            else:
                meshes.append(_wall_segment(
                    [wall_center_perp, center_along, wall_height / 2],
                    [wall_thickness, seg_len, wall_height],
                ))

        # Lintel above door
        door_len = d_max - d_min
        center_along = (d_min + d_max) / 2
        lintel_h = wall_height - d_h
        if lintel_h > 0.001:
            lintel_z = d_h + lintel_h / 2
            if axis == "x":
                meshes.append(_wall_segment(
                    [center_along, wall_center_perp, lintel_z],
                    [door_len, wall_thickness, lintel_h],
                ))
            else:
                meshes.append(_wall_segment(
                    [wall_center_perp, center_along, lintel_z],
                    [wall_thickness, door_len, lintel_h],
                ))

        current_start = d_max

    # Right segment (from last door to wall_end)
    if wall_end - current_start > 0.001:
        seg_len = wall_end - current_start
        center_along = (current_start + wall_end) / 2
        if axis == "x":
            meshes.append(_wall_segment(
                [center_along, wall_center_perp, wall_height / 2],
                [seg_len, wall_thickness, wall_height],
            ))
        else:
            meshes.append(_wall_segment(
                [wall_center_perp, center_along, wall_height / 2],
                [wall_thickness, seg_len, wall_height],
            ))


def generate_structure_mesh(
    layout: Dict[str, Any],
    wall_thickness: float = 0.1,
    floor_thickness: float = 0.05,
    wall_height: float = 2.8,
    door_width: float = 0.9,
    door_height: float = 2.1,
    adjacency_tolerance: float = 0.15,
):
    """
    Generate a structure mesh (floors, walls, doors) from a hierarchical layout.

    Args:
        layout: Hierarchical layout dict with house → rooms
        wall_thickness: Wall thickness in meters
        floor_thickness: Floor thickness in meters
        door_width: Door opening width in meters
        door_height: Door opening height in meters
        adjacency_tolerance: Max gap to consider rooms adjacent

    Returns:
        trimesh.Trimesh: Combined mesh of all structure elements
    """
    import trimesh

    house = layout["house"]
    rooms = house.get("rooms", [])

    if not rooms:
        return trimesh.Trimesh()

    # Find adjacencies and compute door positions
    adjacencies = _find_room_adjacencies(rooms, tolerance=adjacency_tolerance)

    # Build per-room door list
    room_doors = {i: [] for i in range(len(rooms))}

    for adj in adjacencies:
        ri, rj = adj["room_i"], adj["room_j"]
        shared_mid = (adj["shared_min"] + adj["shared_max"]) / 2
        half_door = door_width / 2

        d_min = max(shared_mid - half_door, adj["shared_min"])
        d_max = min(shared_mid + half_door, adj["shared_max"])

        if adj["axis"] == "x":
            # Wall is perpendicular to X, shared range is along Y
            # Room ri has east wall, room rj has west wall
            pi = rooms[ri].get("position", [0, 0, 0])
            si = rooms[ri].get("size", [0, 0, 0])
            pj = rooms[rj].get("position", [0, 0, 0])

            room_doors[ri].append({
                "wall": "east", "door_min": d_min, "door_max": d_max,
                "door_height": min(door_height, wall_height),
            })
            room_doors[rj].append({
                "wall": "west", "door_min": d_min, "door_max": d_max,
                "door_height": min(door_height, wall_height),
            })
        else:
            # Wall perpendicular to Y, shared range along X
            room_doors[ri].append({
                "wall": "north", "door_min": d_min, "door_max": d_max,
                "door_height": min(door_height, wall_height),
            })
            room_doors[rj].append({
                "wall": "south", "door_min": d_min, "door_max": d_max,
                "door_height": min(door_height, wall_height),
            })

    # Ensure every room has at least one door (entrance on exterior wall)
    for i, room in enumerate(rooms):
        if room_doors[i]:
            continue
        pos = room.get("position", [0, 0, 0])
        size = room.get("size", [1, 1, 2.5])
        # Pick the longest wall for the entrance door
        # Prefer south wall (along X) if room is wider, else west wall (along Y)
        if size[0] >= size[1]:
            wall_name = "south"
            wall_min = pos[0]
            wall_max = pos[0] + size[0]
        else:
            wall_name = "west"
            wall_min = pos[1]
            wall_max = pos[1] + size[1]
        wall_mid = (wall_min + wall_max) / 2
        half_door = door_width / 2
        d_min = max(wall_mid - half_door, wall_min)
        d_max = min(wall_mid + half_door, wall_max)
        room_doors[i].append({
            "wall": wall_name, "door_min": d_min, "door_max": d_max,
            "door_height": min(door_height, wall_height),
        })

    # Generate meshes
    all_meshes = []

    for i, room in enumerate(rooms):
        pos = room.get("position", [0, 0, 0])
        size = list(room.get("size", [1, 1, 2.5]))
        size[2] = wall_height

        # Floor
        all_meshes.append(_create_room_floor(pos, size, floor_thickness))

        # Walls with door cuts
        wall_meshes = _create_room_walls(pos, size, wall_thickness, room_doors[i])
        all_meshes.extend(wall_meshes)

    return trimesh.util.concatenate(all_meshes)


def main():
    """Command-line interface for hierarchical layout utilities."""
    import argparse

    parser = argparse.ArgumentParser(description="Utilities for hierarchical layouts")
    parser.add_argument(
        "command",
        choices=["validate", "stats", "tree", "flatten", "all"],
        help="Command to execute",
    )
    parser.add_argument("input_file", help="Input JSON file (hierarchical layout)")
    parser.add_argument("-o", "--output", help="Output file for flatten command")

    args = parser.parse_args()

    # Load layout
    with open(args.input_file, "r") as f:
        layout = json.load(f)

    if args.command == "validate":
        is_valid, errors = validate_hierarchical_layout(layout)
        if is_valid:
            print("✓ Layout is valid!")
        else:
            print("✗ Layout has errors:")
            for error in errors:
                print(f"  - {error}")
            sys.exit(1)

    elif args.command == "stats":
        print_layout_stats(layout)

    elif args.command == "tree":
        print_layout_tree(layout)

    elif args.command == "flatten":
        flat = hierarchical_to_flat(layout)

        if args.output:
            with open(args.output, "w") as f:
                json.dump(flat, f, indent=2)
            print(f"Flattened layout saved to {args.output}")
        else:
            print(json.dumps(flat, indent=2))

    elif args.command == "all":
        print("\n")
        print_layout_tree(layout)
        print()
        print_layout_stats(layout)
        print()
        is_valid, errors = validate_hierarchical_layout(layout)
        if is_valid:
            print("✓ Layout is valid!")
        else:
            print("✗ Layout has errors:")
            for error in errors:
                print(f"  - {error}")


if __name__ == "__main__":
    main()
