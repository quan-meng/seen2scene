# Hierarchical 3D Scene Layout Generation

## Overview

This hierarchical layout generator creates 3D scenes from coarse to fine detail:

```
House → Rooms → Furniture → Items
```

## Key Features

### 1. **Hierarchical Structure**
- **House level**: Defines overall space and room placement
- **Room level**: Furniture placement within each room
- **Furniture level**: Small items placed on furniture surfaces

### 2. **Local Coordinate Systems**
Each level uses its parent's coordinate system:
- Rooms use house origin (0, 0, 0)
- Furniture uses room origin
- Items use furniture origin

This makes spatial reasoning easier at each level.

### 3. **LLM-Driven Item Placement**
The LLM autonomously decides whether items should be placed on furniture:
- **First**: LLM evaluates if furniture typically has items (tables ✓, chairs ✗)
- **Then**: If appropriate, generates realistic items based on room context
- **No hardcoded lists**: Decisions based on furniture type, room type, and common sense
- **Contextual**: Kitchen counter gets kitchen items, desk gets office items

### 4. **Dual Output Format**
Generates two files:
- `house_hierarchical_XXXX.json`: Full hierarchical structure
- `house_flat_XXXX.json`: Flattened to world coordinates (compatible with existing code)

## Usage

```bash
# Basic usage - generate 1 house with 3 rooms
python models/layout/get_layout_hierarchical.py --num 1 --num_rooms 3

# Generate 5 houses with custom size and more rooms
python models/layout/get_layout_hierarchical.py \
  --num 5 \
  --house_size 10.0 10.0 2.8 \
  --num_rooms 4 \
  --verbose

# Use a specific model
python models/layout/get_layout_hierarchical.py \
  --num 1 \
  --num_rooms 3 \
  --model gpt-4o

# Control maximum items per furniture
python models/layout/get_layout_hierarchical.py \
  --num 1 \
  --num_rooms 3 \
  --max_items_per_furniture 3
```

## Output Format

### Hierarchical Format
```json
{
  "house": {
    "size": [8.0, 8.0, 2.8],
    "rooms": [
      {
        "type": "bedroom",
        "position": [0.0, 0.0, 0.0],
        "size": [4.0, 4.0, 2.8],
        "furniture": [
          {
            "name": "bed",
            "bbox": [[0.5, 0.5, 0.0], [2.5, 2.3, 0.6]],
            "items": [
              {
                "name": "pillow",
                "bbox": [[0.2, 0.2, 0.6], [0.5, 0.4, 0.75]]
              },
              {
                "name": "lamp",
                "bbox": [[2.0, 0.2, 0.6], [2.2, 0.4, 0.9]]
              }
            ]
          },
          {
            "name": "nightstand",
            "bbox": [[2.7, 0.5, 0.0], [3.2, 1.0, 0.5]],
            "items": [
              {
                "name": "clock",
                "bbox": [[0.1, 0.1, 0.5], [0.3, 0.3, 0.6]]
              }
            ]
          }
        ]
      },
      {
        "type": "living_room",
        "position": [4.0, 0.0, 0.0],
        "size": [4.0, 4.0, 2.8],
        "furniture": [
          {
            "name": "sofa",
            "bbox": [[0.5, 0.5, 0.0], [2.5, 1.5, 0.8]],
            "items": []
          },
          {
            "name": "coffee_table",
            "bbox": [[1.0, 2.0, 0.0], [2.5, 3.0, 0.4]],
            "items": [
              {
                "name": "book",
                "bbox": [[0.2, 0.2, 0.4], [0.4, 0.5, 0.45]]
              }
            ]
          }
        ]
      }
    ]
  }
}
```

### Flat Format (for compatibility)
```json
{
  "object_names": [
    "bed",
    "pillow",
    "lamp",
    "nightstand",
    "clock",
    "sofa",
    "coffee_table",
    "book"
  ],
  "object_bboxes": [
    [0.5, 0.5, 0.0, 2.5, 2.3, 0.6],      // bed (world coords)
    [0.7, 0.7, 0.6, 1.0, 0.9, 0.75],     // pillow (world coords)
    [2.5, 0.7, 0.6, 2.7, 0.9, 0.9],      // lamp (world coords)
    [2.7, 0.5, 0.0, 3.2, 1.0, 0.5],      // nightstand (world coords)
    [2.8, 0.6, 0.5, 3.0, 0.8, 0.6],      // clock (world coords)
    [4.5, 0.5, 0.0, 6.5, 1.5, 0.8],      // sofa (world coords)
    [5.0, 2.0, 0.0, 6.5, 3.0, 0.4],      // coffee_table (world coords)
    [5.2, 2.2, 0.4, 5.4, 2.5, 0.45]      // book (world coords)
  ]
}
```

## Comparison with Flat Layout Generation

| Feature | Flat (`get_layout.py`) | Hierarchical (`get_layout_hierarchical.py`) |
|---------|------------------------|---------------------------------------------|
| Structure | Single-level list | Multi-level hierarchy |
| Coordinate System | Global (world coords) | Local (parent-relative) |
| Spatial Reasoning | Complex (all objects) | Simpler (per-level) |
| Item Placement | Manual filtering | Automatic (furniture-aware) |
| Scalability | Limited (flat list) | Better (hierarchical) |
| Output | Single format | Dual (hierarchical + flat) |
| Use Case | Simple rooms/patches | Complex houses |

## How It Works

### Step 1: House-level Generation
```
Input: House size (8×8×2.8m), number of rooms (3)
Prompt: "Generate 3 rooms for a house..."
Output: Room types, positions, and sizes
```

### Step 2: Room-level Generation (for each room)
```
Input: Room type (bedroom), room size (4×4×2.8m)
Prompt: "Generate furniture for a bedroom..."
Output: Furniture items with local coordinates
```

### Step 3: Furniture-level Generation (for each furniture)
```
Input: Furniture type (table), furniture size, surface height
Prompt: "Place items ON TOP of a table..."
Output: Items with local coordinates (relative to furniture)
```

### Step 4: Coordinate Transformation
Convert local coordinates to world coordinates:
```python
world_pos = house_origin + room_pos + furniture_pos + item_pos
```

## Advantages

1. **Easier Spatial Reasoning**: Each level works in its own coordinate space
2. **More Realistic**: Hierarchical structure mirrors real-world organization
3. **Automatic Constraints**: Items automatically placed on surfaces, not floating
4. **Scalable**: Can generate large houses without overwhelming the model
5. **Semantic Structure**: Preserves relationships (which items belong to which furniture)

## Prerequisites

Same as flat layout generation:
```bash
npm install -g @openai/codex
codex login
```

## Customization

### Modify Room Types
Edit `ROOM_TYPES` in the script:
```python
ROOM_TYPES = [
    "bedroom", "living_room", "kitchen", "bathroom",
    "your_custom_room_type"
]
```

### Customize Item Placement Behavior
The LLM decides which furniture gets items based on the prompt in `generate_furniture_items_prompt()`.
You can customize this by:

1. **Adjust the prompt logic** to guide LLM decisions
2. **Add context hints** (e.g., "minimalist style = fewer items")
3. **Modify max_items_per_furniture** parameter to control density
4. **Edit available categories** in `valid_categories` to limit/expand item types

Example: Add style preferences to the prompt:
```python
prompt = f"""...
CONTEXT:
- Room type: {room_type}
- Furniture: {furniture_name}
- Style: minimalist  # <-- Add style hint
...
```

## Tips

1. **Start small**: Begin with `--num_rooms 2` to test the system
2. **Adjust house size**: Larger houses need more rooms to look natural
3. **Use verbose mode**: `--verbose` to see generation progress
4. **Check both outputs**: Hierarchical for structure, flat for rendering
5. **Iterate on failures**: Some generations may fail; increase `--num` to get more successes

## Future Improvements

- [ ] Support for multi-story houses (different Z levels)
- [ ] Wall and door generation
- [ ] More sophisticated item placement (stacking, grouping)
- [ ] Physics-based validation (no floating objects)
- [ ] Style transfer (modern, vintage, minimalist, etc.)
