# Quick Start: Hierarchical Layout Generation

## Installation

```bash
# Install Codex CLI
npm install -g @openai/codex

# Login to Codex
codex login
```

## Generate Your First Hierarchical Layout

```bash
# Generate 1 house with 3 rooms
python models/layout/get_layout_hierarchical.py --num 1 --num_rooms 3 --verbose

# Output will be in: logs/generated_layouts_hierarchical/
# - house_hierarchical_0000.json (hierarchical structure)
# - house_flat_0000.json (flattened for compatibility)
```

## Inspect the Generated Layout

```bash
# View as tree
python models/layout/hierarchical_utils.py tree logs/generated_layouts_hierarchical/house_hierarchical_0000.json

# View statistics
python models/layout/hierarchical_utils.py stats logs/generated_layouts_hierarchical/house_hierarchical_0000.json

# Validate layout
python models/layout/hierarchical_utils.py validate logs/generated_layouts_hierarchical/house_hierarchical_0000.json

# See everything
python models/layout/hierarchical_utils.py all logs/generated_layouts_hierarchical/house_hierarchical_0000.json
```

## Example Output

### Tree View
```
🏠 House (8.0×8.0×2.8m)
  📦 Room 1: bedroom @ (0.0, 0.0) [4.0×4.0×2.8m]
    🪑 bed @ (0.5, 0.5) [2.00×1.80×0.60m] (2 items)
      📌 pillow @ (0.20, 0.20, 0.60)
      📌 lamp @ (1.80, 0.20, 0.60)
    🪑 nightstand @ (2.7, 0.5) [0.50×0.50×0.50m] (2 items)
      📌 clock @ (0.10, 0.10, 0.50)
      📌 book @ (0.30, 0.20, 0.50)
```

### Statistics
```
Rooms:     3
Furniture: 11
Items:     14

Furniture with items:    7 (63.6%)
Furniture without items: 4 (36.4%)
Avg items per furniture: 1.27

Rooms by type:
  bedroom: 1
  kitchen: 1
  living_room: 1
```

## Understanding the Coordinate Systems

### 1. House Level (World Coordinates)
```python
House origin: (0, 0, 0)
House size: (8, 8, 2.8) meters
```

### 2. Room Level (Relative to House)
```python
Room position: (4, 0, 0)  # Relative to house origin
Room size: (4, 4, 2.8)    # Room dimensions
```

### 3. Furniture Level (Relative to Room)
```python
Furniture bbox: [[0.5, 0.5, 0.0], [2.5, 2.3, 0.6]]
# Relative to room origin (0, 0, 0)

World position = Room position + Furniture position
               = (4, 0, 0) + (0.5, 0.5, 0.0)
               = (4.5, 0.5, 0.0)
```

### 4. Item Level (Relative to Furniture)
```python
Item bbox: [[0.2, 0.2, 0.6], [0.5, 0.4, 0.75]]
# Relative to furniture origin (0, 0, 0)

World position = Room position + Furniture position + Item position
               = (4, 0, 0) + (0.5, 0.5, 0.0) + (0.2, 0.2, 0.6)
               = (4.7, 0.7, 0.6)
```

## Common Options

### Generate Multiple Houses
```bash
python models/layout/get_layout_hierarchical.py --num 5 --num_rooms 3
```

### Custom House Size
```bash
python models/layout/get_layout_hierarchical.py \
  --num 1 \
  --house_size 10.0 10.0 2.8 \
  --num_rooms 4
```

### Use Specific AI Model
```bash
python models/layout/get_layout_hierarchical.py \
  --num 1 \
  --num_rooms 3 \
  --model gpt-4o
```

### Control Item Density
```bash
python models/layout/get_layout_hierarchical.py \
  --num 1 \
  --num_rooms 3 \
  --max_items_per_furniture 3
```

### Custom Output Directory
```bash
python models/layout/get_layout_hierarchical.py \
  --num 1 \
  --num_rooms 3 \
  --out_dir my_layouts \
  --prefix my_house
```

## Convert Existing Flat Layout

If you have a flat layout, you can still inspect it:

```bash
# The hierarchical utilities expect hierarchical format
# To convert flat → hierarchical (requires manual clustering)
# For now, use the hierarchical generator directly
```

## Troubleshooting

### "Not logged in to Codex CLI"
```bash
codex login
```

### Generation Fails
- Check your internet connection
- Try with `--verbose` to see detailed errors
- Increase `--num` to generate multiple attempts
- Try a different `--model`

### Invalid Layout
```bash
# Validate the layout
python models/layout/hierarchical_utils.py validate path/to/layout.json

# This will show specific errors like:
# - Objects extending beyond bounds
# - Missing required fields
# - Invalid coordinate values
```

## Next Steps

1. **Visualize in 3D**: Import the flat layout into Blender or your rendering engine
2. **Batch Generation**: Create a dataset by generating many layouts
3. **Customize Categories**: Edit `ROOM_TYPES`, `FURNITURE_WITH_ITEMS`, and `SURFACE_ITEMS` in the script
4. **Integrate with Pipeline**: Use the hierarchical structure for better scene understanding

## File Structure

```
models/layout/
├── get_layout_hierarchical.py       # Main generation script
├── hierarchical_utils.py            # Validation and conversion utilities
├── README_hierarchical.md           # Detailed documentation
├── QUICKSTART_hierarchical.md       # This file
└── example_hierarchical_layout.json # Example output

logs/generated_layouts_hierarchical/
├── house_hierarchical_0000.json     # Generated hierarchical layout
└── house_flat_0000.json             # Generated flat layout
```

## Key Differences from Flat Generation

| Aspect | Flat (`get_layout.py`) | Hierarchical (`get_layout_hierarchical.py`) |
|--------|------------------------|---------------------------------------------|
| Generation | Single LLM call | Multiple calls (house → rooms → furniture → items) |
| Structure | Flat list | Tree structure |
| Coordinates | World coords | Parent-relative |
| Items | Mixed with furniture | Explicitly on furniture surfaces |
| Scalability | Limited | High (decomposed problem) |

## Example: Complete Workflow

```bash
# 1. Generate layout
python models/layout/get_layout_hierarchical.py \
  --num 1 \
  --num_rooms 3 \
  --house_size 8.0 8.0 2.8 \
  --verbose

# 2. Validate it
python models/layout/hierarchical_utils.py validate \
  logs/generated_layouts_hierarchical/house_hierarchical_0000.json

# 3. View statistics
python models/layout/hierarchical_utils.py stats \
  logs/generated_layouts_hierarchical/house_hierarchical_0000.json

# 4. View as tree
python models/layout/hierarchical_utils.py tree \
  logs/generated_layouts_hierarchical/house_hierarchical_0000.json

# 5. Use the flat version for rendering
# logs/generated_layouts_hierarchical/house_flat_0000.json
```

## Tips for Best Results

1. **Start small**: Use 2-3 rooms first to test
2. **Reasonable sizes**: Don't make houses too large (> 12×12m gets sparse)
3. **Verbose mode**: Use `--verbose` to understand generation progress
4. **Multiple attempts**: Some generations fail; use `--num 5` to get several attempts
5. **Validate always**: Always run validation before using layouts

## Support

- See `README_hierarchical.md` for detailed documentation
- Check `example_hierarchical_layout.json` for structure reference
- Use `--verbose` flag to debug generation issues
