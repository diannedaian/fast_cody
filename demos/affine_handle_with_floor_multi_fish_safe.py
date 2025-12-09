import sys
import os
import numpy as np

# Add the source directory to the path so we can import directly from source
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from fast_cody.apps.interactive_cd_affine_handle_multi_fish import interactive_cd_affine_handle_multi_fish
import fast_cody as fc

# Start with just 2 copies of the default fish to test the fix
# This avoids issues with missing custom meshes
default_fish_msh = fc.get_data("./cd_fish.msh")
default_fish_texture_png = fc.get_data("./cd_fish_tex.png")
default_fish_texture_obj = fc.get_data("./cd_fish_tex.obj")

# Create lists for 2 fishes - both using default fish
msh_files = [
    default_fish_msh,      # Fish 0: default
    default_fish_msh,      # Fish 1: default (duplicate)
]

texture_png_list = [
    default_fish_texture_png,  # Fish 0: default texture
    default_fish_texture_png,  # Fish 1: default texture
]

texture_obj_list = [
    default_fish_texture_obj,  # Fish 0: default texture obj
    default_fish_texture_obj,  # Fish 1: default texture obj
]

# Initialize fish positions
fish_positions = [
    np.array([-2.0, 0.0, 0.0]),  # Fish 0: left
    np.array([2.0, 0.0, 0.0]),   # Fish 1: right
]

print("Starting with 2 fishes to test stability...")
print("If this works, you can gradually increase to 7 fishes")

interactive_cd_affine_handle_multi_fish(
    num_fishes=2,
    msh_files=msh_files,
    texture_png_list=texture_png_list,
    texture_obj_list=texture_obj_list,
    fish_positions=fish_positions,
    enable_caustics=True,
    secondary_motion_scale=0.9  # Scale secondary motion to 90% strength (more visible tail wagging)
    # Note: Fish 0 and 1 (default fish) have no secondary motion clamping
)
