from fast_cody.apps import interactive_cd_affine_handle
from fast_cody import get_data
import fast_cd_pyb as fcdp
import fast_cody as fcd
import os


# name = "./data/dromedary/dromedary.msh"
# interactive_cd_affine_handle(name, read_cache=False, num_modes=16)


# name = "./data/xyz_dragon/xyz_dragon.msh"
# interactive_cd_affine_handle(name,  read_cache=False, num_modes=16)

# name = get_data("./elephant/elephant.msh")
# interactive_cd_affine_handle(name, rho=1000, read_cache=False, num_modes=16)

# name = "./data/elephant/elephant.msh"
# [V, F, T] = fcd.read_msh(name)
# interactive_cd_affine_handle(V=V, T=T)
#
# name = get_data("./king_ghidora/king_ghidora.msh")
# texture_obj = get_data("./king_ghidora/king_ghidora_tex.obj")
# texture_png = get_data("./king_ghidora/king_ghidora_tex.png")
# interactive_cd_affine_handle(name, texture_png=texture_png,
#                              texture_obj=texture_obj, read_cache=True)
#
#
# name = get_data("./bulldog/bulldog.msh")
# texture_obj = get_data("./bulldog/bulldog_tex.obj")
# texture_png = get_data("./bulldog/bulldog_tex.png")
# interactive_cd_affine_handle(name, mu = 5e3, rho=1000, read_cache=False, num_modes=16,
#                              texture_obj=texture_obj, texture_png=texture_png,
#                              constraint_enforcement="optimal")
#
# #
# Use the project's data directory, not the package's data directory
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# # cd_fish (commented out)
# name = os.path.join(project_root, "data", "cd_fish", "cd_fish.msh")
# texture_png = os.path.join(project_root, "data", "cd_fish", "cd_fish_tex.png")
# texture_obj = os.path.join(project_root, "data", "cd_fish", "cd_fish_tex.obj")
# interactive_cd_affine_handle(name, texture_png=texture_png, texture_obj=texture_obj,
#                              read_cache=False, num_modes=16, constraint_enforcement="optimal")

# giant_squid
# name = os.path.join(project_root, "data", "giant_squid", "giant_squid.msh")
# texture_png = os.path.join(project_root, "data", "giant_squid", "giant_squid_tex.png")
# texture_obj = os.path.join(project_root, "data", "giant_squid", "giant_squid_tex.obj")
# interactive_cd_affine_handle(name, texture_png=texture_png, texture_obj=texture_obj,
#                              read_cache=False, num_modes=16, constraint_enforcement="optimal")

# Test generated model from pipeline
name = os.path.join(project_root, "outputs", "converted", "model", "model.msh")
texture_png = os.path.join(project_root, "outputs", "converted", "model", "texture.png")
texture_obj = os.path.join(project_root, "outputs", "converted", "model", "model.obj")
interactive_cd_affine_handle(name, texture_png=texture_png, texture_obj=texture_obj,
                             read_cache=False, num_modes=16, constraint_enforcement="optimal")
