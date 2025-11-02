from fast_cody.apps import interactive_cd_rig_anim
import fast_cody as fcd
import os

# Use the project's data directory, not the package's data directory
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

## Test input msh_file
msh_file = os.path.join(project_root, "data", "cd_fish", "cd_fish.msh")
# fcd.apps.interactive_cd_rig_anim(msh_file)


# Test input V and T
[V, F, T] = fcd.read_msh(msh_file)
# fcd.apps.interactive_cd_rig_anim(V=V, T=T)


# Test input Wp and P0
import igl
import numpy as np
rig_file = os.path.join(project_root, "data", "cd_fish", "rigs", "skeleton_rig", "skeleton_rig.json")
[Vpsurf, Fpsurf, Wpsurface, P0, lengths, pI] = fcd.read_rig_from_json(rig_file)

# Use vertex indices as points (original approach - computes distances to vertices)
aI = np.arange(V.shape[0]).reshape(-1, 1).astype(np.int32)
[D2, bI, CP] = igl.point_mesh_squared_distance(Vpsurf, V, aI)
Wp = fcd.diffuse_weights(V, T, Wpsurface, bI, dt=10000)

[V, F, T] = fcd.read_msh(msh_file)
# fcd.apps.interactive_cd_rig_anim(V=V, T=T, Wp=Wp, P0=P0)


# Test input P
anim_file = os.path.join(project_root, "data", "cd_fish", "rigs", "skeleton_rig", "anim", "swim.json")
P = fcd.read_rig_anim_from_json(anim_file)
interactive_cd_rig_anim(V=V, T=T, Wp=Wp, P0=2*P0, P=2*P)
