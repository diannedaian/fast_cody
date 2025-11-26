# Caustics Textures

This folder contains caustics texture files for underwater visual effects that simulate light refracting through a water surface.

## Animated Caustics Atlas

The `caustics_atlas.png` file is a horizontal texture atlas containing 16 frames of animated caustics:
- **Dimensions**: 4096 x 256 (16 frames of 256x256 each)
- **Format**: PNG with alpha channel
- **Usage**: UV coordinates are animated to cycle through frames horizontally

### Building the Atlas

To rebuild the atlas from individual frames, run:

```bash
cd /path/to/fast_cody/src/fast_cody/data
python build_caustics_atlas.py
```

This script:
1. Loads all `caust_*.png` files from this directory (16 frames)
2. Concatenates them horizontally into one atlas
3. Saves as `caustics_atlas.png`

## Usage in Code

Access caustics textures in your code using:

```python
import fast_cody as fc

# Get the animated caustics atlas
caustics_atlas_path = fc.get_data("caustics/caustics_atlas.png")

# For individual frames (if needed)
caustics_frame = fc.get_data("caustics/caust_001.png")
```

## Caustics Animation

The `interactive_cd_affine_handle_with_caustics()` function demonstrates animated caustics:
- Cycles through 16 frames in the atlas
- UV coordinates shift horizontally to show different frames
- Creates realistic underwater light patterns on the ocean floor

## File Organization

- `caust_001.png` through `caust_016.png`: Individual caustics frames (256x256 each)
- `caustics_atlas.png`: Combined horizontal atlas (4096x256)
- `caustics_atlas_noalpha.png`: Atlas without alpha channel (if needed)
- `build_caustics_atlas.py`: Script to rebuild the atlas
