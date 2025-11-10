import sys
import os

# # Add the source directory to the path so we can import directly from source
# sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from fast_cody.apps.interactive_cd_dual_mode import interactive_cd_dual_mode

interactive_cd_dual_mode()
# fc.apps.interactive_cd_dual_mode(
#     msh_file="./data/giant_squid/giant_squid.msh",
#     texture_png="./data/giant_squid/giant_squid_tex.png",
#     texture_obj="./data/giant_squid/giant_squid_tex.obj",
#     read_cache=False,
#     num_modes=16,
#     constraint_enforcement="optimal",
# )