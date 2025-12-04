import sys
import os
import numpy as np

# Add the source directory to the path so we can import directly from source
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from fast_cody.apps.interactive_cd_affine_handle_multi_fish import interactive_cd_affine_handle_multi_fish
import fast_cody as fc

# Set up paths for 3 fishes
# Fish 0 uses default fish, Fish 1 uses model from 20251121_122631, Fish 2 uses model from 20251121_121348
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
new_model_dir_1 = os.path.join(project_root, "outputs", "converted", "20251121_122631")
new_model_msh_1 = os.path.join(new_model_dir_1, "model.msh")
new_model_texture_png_1 = os.path.join(new_model_dir_1, "texture.png")
new_model_texture_obj_1 = os.path.join(new_model_dir_1, "model.obj")

new_model_dir_2 = os.path.join(project_root, "outputs", "converted", "20251121_121348")
new_model_msh_2 = os.path.join(new_model_dir_2, "model.msh")
new_model_texture_png_2 = os.path.join(new_model_dir_2, "texture.png")
new_model_texture_obj_2 = os.path.join(new_model_dir_2, "model.obj")

new_model_dir_3 = os.path.join(project_root, "outputs", "converted", "20251202_200313_1387598517346107392")
new_model_msh_3 = os.path.join(new_model_dir_3, "model.msh")
new_model_texture_png_3 = os.path.join(new_model_dir_3, "texture.png")
new_model_texture_obj_3 = os.path.join(new_model_dir_3, "model.obj")

# Default fish paths
default_fish_msh = fc.get_data("./cd_fish.msh")
default_fish_texture_png = fc.get_data("./cd_fish_tex.png")
default_fish_texture_obj = fc.get_data("./cd_fish_tex.obj")

# Create lists for 8 fishes: [fish0, fish1, fish2, fish3, fish4, fish5, fish6, fish7]
# Fish 0: default, Fish 1,3,4,5: model from 20251121_122631, Fish 2,6: model from 20251121_121348, Fish 7: model from 20251202_200313_1387598517346107392
msh_files = [
    default_fish_msh,      # Fish 0: default
    new_model_msh_1,       # Fish 1: new model from 20251121_122631
    new_model_msh_2,       # Fish 2: new model from 20251121_121348
    new_model_msh_1,       # Fish 3: new model from 20251121_122631 (duplicate of fish 1)
    new_model_msh_1,       # Fish 4: new model from 20251121_122631 (duplicate of fish 1)
    new_model_msh_1,       # Fish 5: new model from 20251121_122631 (duplicate of fish 1)
    new_model_msh_2,       # Fish 6: new model from 20251121_121348 (duplicate of fish 2)
    new_model_msh_3        # Fish 7: new model from 20251202_200313_1387598517346107392
]

texture_png_list = [
    default_fish_texture_png,  # Fish 0: default texture
    new_model_texture_png_1,   # Fish 1: new texture from 20251121_122631
    new_model_texture_png_2,   # Fish 2: new texture from 20251121_121348
    new_model_texture_png_1,   # Fish 3: new texture from 20251121_122631
    new_model_texture_png_1,   # Fish 4: new texture from 20251121_122631
    new_model_texture_png_1,   # Fish 5: new texture from 20251121_122631
    new_model_texture_png_2,   # Fish 6: new texture from 20251121_121348
    new_model_texture_png_3    # Fish 7: new texture from 20251202_200313_1387598517346107392
]

texture_obj_list = [
    default_fish_texture_obj,  # Fish 0: default texture obj
    new_model_texture_obj_1,   # Fish 1: new texture obj from 20251121_122631
    new_model_texture_obj_2,   # Fish 2: new texture obj from 20251121_121348
    new_model_texture_obj_1,   # Fish 3: new texture obj from 20251121_122631
    new_model_texture_obj_1,   # Fish 4: new texture obj from 20251121_122631
    new_model_texture_obj_1,   # Fish 5: new texture obj from 20251121_122631
    new_model_texture_obj_2,   # Fish 6: new texture obj from 20251121_121348
    new_model_texture_obj_3    # Fish 7: new texture obj from 20251202_200313_1387598517346107392
]

# Run with 8 fishes, caustics, and floor
# Fish 0: default, Fish 1,3,4,5: model from 20251121_122631, Fish 2,6: model from 20251121_121348, Fish 7: model from 20251202_200313_1387598517346107392
# Initialize each fish at specific positions
# Note: Each fish will have its own cache directory (fish_0, fish_1, etc.) to avoid sharing secondary motion cache
fish_positions = [
    np.array([-5.5701137, 5.0341086, 0.0]),        # Fish 0
    np.array([0.06162413, 0.08752766, -4.3016896]), # Fish 1
    np.array([4.321723, 3.3223279, -6.2218714]),   # Fish 2
    np.array([-2.0, 2.0, -2.0]),                   # Fish 3
    np.array([2.0, 2.0, -2.0]),                    # Fish 4 (new position)
    np.array([0.0, 4.0, -4.0]),                    # Fish 5 (new position)
    np.array([0.0, 4.0, -2.0]),                    # Fish 6: to the right top of fish 3 (fish 3 is at [-2.0, 2.0, -2.0])
    np.array([-2.00, 0.383, 0.72])                  # Fish 7: new model from 20251202_200313_1387598517346107392
]

interactive_cd_affine_handle_multi_fish(
    num_fishes=8,
    msh_files=msh_files,
    texture_png_list=texture_png_list,
    texture_obj_list=texture_obj_list,
    fish_positions=fish_positions,
    enable_caustics=True
)
