import sys
import os
import numpy as np

# Add the source directory to the path so we can import directly from source
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from fast_cody.apps.interactive_cd_affine_handle_multi_fish import interactive_cd_affine_handle_multi_fish
import fast_cody as fc

# Use available fish meshes from the data directory
# Mix of different species for variety
default_fish_msh = fc.get_data("./cd_fish.msh")
default_fish_texture_png = fc.get_data("./cd_fish_tex.png")
default_fish_texture_obj = fc.get_data("./cd_fish_tex.obj")

# Check which alternative meshes exist
dolphin_msh = fc.get_data("./dolphin/dolphin.msh")
beta_fish_msh = fc.get_data("./beta_fish/beta_fish.msh")

# Create lists for 7 fishes using available meshes
# Pattern: default, dolphin, beta, default, dolphin, beta, default
msh_files = [
    default_fish_msh,      # Fish 0: default cd_fish
    default_fish_msh,      # Fish 1: default cd_fish
    default_fish_msh,      # Fish 2: default cd_fish
    default_fish_msh,      # Fish 3: default cd_fish
    default_fish_msh,      # Fish 4: default cd_fish
    default_fish_msh,      # Fish 5: default cd_fish
    default_fish_msh       # Fish 6: default cd_fish
]

texture_png_list = [
    default_fish_texture_png,  # Fish 0
    default_fish_texture_png,  # Fish 1
    default_fish_texture_png,  # Fish 2
    default_fish_texture_png,  # Fish 3
    default_fish_texture_png,  # Fish 4
    default_fish_texture_png,  # Fish 5
    default_fish_texture_png   # Fish 6
]

texture_obj_list = [
    default_fish_texture_obj,  # Fish 0
    default_fish_texture_obj,  # Fish 1
    default_fish_texture_obj,  # Fish 2
    default_fish_texture_obj,  # Fish 3
    default_fish_texture_obj,  # Fish 4
    default_fish_texture_obj,  # Fish 5
    default_fish_texture_obj   # Fish 6
]

# Position fishes in an interesting underwater formation
fish_positions = [
    np.array([-5.5701137, 5.0341086, 0.0]),        # Fish 0: top left
    np.array([0.06162413, 0.08752766, -4.3016896]), # Fish 1: center front
    np.array([4.321723, 3.3223279, -6.2218714]),   # Fish 2: right back
    np.array([-2.0, 2.0, -2.0]),                   # Fish 3: left mid
    np.array([2.0, 2.0, -2.0]),                    # Fish 4: right mid
    np.array([0.0, 4.0, -4.0]),                    # Fish 5: center high back
    np.array([0.0, 4.0, -2.0])                     # Fish 6: center high front
]

print("="*80)
print("Starting 7-Fish Underwater Scene Demo")
print("="*80)
print("Features:")
print("  - 7 animated fish with independent controls")
print("  - Underwater visual effects (gradient + overlay)")
print("  - Animated caustics on ocean floor")
print("  - 5x5 ocean floor grid")
print()
print("Controls:")
print("  1-7: Switch between fish")
print("  g:   Toggle transform mode (translate/rotate/scale)")
print("="*80)
print()

interactive_cd_affine_handle_multi_fish(
    num_fishes=7,
    msh_files=msh_files,
    texture_png_list=texture_png_list,
    texture_obj_list=texture_obj_list,
    fish_positions=fish_positions,
    enable_caustics=True,
    secondary_motion_scale=0.8  # Scale secondary motion to 90% strength (more visible tail wagging
)
