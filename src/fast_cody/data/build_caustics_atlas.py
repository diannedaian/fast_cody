#!/usr/bin/env python3
"""
Build a caustics texture atlas by concatenating all PNG files horizontally.

This script:
1. Loads all PNG files from ./caustics/ directory
2. Ensures they are all the same size
3. Concatenates them horizontally into one large atlas image
4. Saves the result as "caustics_atlas.png" in the data folder
"""

import os
from pathlib import Path
from PIL import Image
import numpy as np

def build_caustics_atlas():
    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    caustics_dir = script_dir / "caustics"
    output_path = script_dir / "caustics_atlas.png"

    # Check if caustics directory exists
    if not caustics_dir.exists():
        print(f"Error: Caustics directory not found: {caustics_dir}")
        return

    # Find all PNG files in caustics directory, excluding atlas files
    all_png_files = sorted(caustics_dir.glob("*.png"))
    png_files = [f for f in all_png_files if not f.name.startswith("caustics_atlas")]

    if len(png_files) == 0:
        print(f"Error: No PNG files found in {caustics_dir} (excluding atlas files)")
        return

    print(f"Found {len(png_files)} PNG file(s):")
    for png_file in png_files:
        print(f"  - {png_file.name}")

    # Load all images
    images = []
    first_size = None

    for png_file in png_files:
        img = Image.open(png_file)
        img_array = np.array(img)

        # Get image dimensions
        if img.mode == 'RGBA':
            height, width = img_array.shape[:2]
        else:
            height, width = img_array.shape[:2]

        # Check if all images have the same size
        if first_size is None:
            first_size = (width, height)
            first_mode = img.mode
        else:
            if (width, height) != first_size:
                raise ValueError(
                    f"Image {png_file.name} has size {width}x{height}, "
                    f"but expected {first_size[0]}x{first_size[1]}"
                )
            if img.mode != first_mode:
                # Convert to RGBA if modes differ
                if first_mode == 'RGBA':
                    img = img.convert('RGBA')
                elif img.mode == 'RGBA':
                    first_mode = 'RGBA'
                    # Convert previous images to RGBA
                    images = [im.convert('RGBA') for im in images]
                    img = img.convert('RGBA')

        images.append(img)

    # Get dimensions
    frame_width, frame_height = first_size
    num_frames = len(images)
    atlas_width = frame_width * num_frames
    atlas_height = frame_height

    # Create the atlas by concatenating horizontally
    atlas = Image.new(first_mode, (atlas_width, atlas_height))

    x_offset = 0
    for img in images:
        atlas.paste(img, (x_offset, 0))
        x_offset += frame_width

    # Save the atlas
    atlas.save(output_path)

    # Print statistics
    print("\n" + "=" * 60)
    print("CAUSTICS ATLAS BUILT SUCCESSFULLY")
    print("=" * 60)
    print(f"Number of frames:     {num_frames}")
    print(f"Frame dimensions:     {frame_width} x {frame_height}")
    print(f"Atlas dimensions:      {atlas_width} x {atlas_height}")
    print(f"Output file:          {output_path}")
    print("=" * 60)

if __name__ == "__main__":
    build_caustics_atlas()
