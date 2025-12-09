#!/usr/bin/env python3
"""
Minimal test to reproduce the segfault with the exact same structure as the main function
"""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import fast_cody as fc
import fast_cd_pyb as fcd
import scipy as sp

print("Loading meshes...")
msh_file = fc.get_data("./cd_fish.msh")
[V1, _, T1] = fcd.readMSH(msh_file)
[V2, _, T2] = fcd.readMSH(msh_file)  # Same mesh, different instance

Vs = [V1, V2]
Ts = [T1, T2]

num_modes = 16
num_clusters = 100
mu = 1e4
rho = 1e3

fishes = []

for fish_idx in range(2):
    print(f"\n{'='*60}")
    print(f"Creating fish {fish_idx + 1}/2...")
    print(f"{'='*60}")

    import gc
    import time
    gc.collect()
    time.sleep(0.5)

    print(f"  Loading mesh data...")
    V = Vs[fish_idx].copy()
    T = Ts[fish_idx].copy()

    print(f"  Scaling and centering geometry...")
    [V_centered, so, to] = fcd.scale_and_center_geometry(V, 1, np.array([[0, 0, 0.]]))

    print(f"  Creating handle...")
    Wp = np.ones((V_centered.shape[0], 1))
    J = fc.lbs_jacobian(V_centered, Wp)

    fish_cache_dir = f"./cache/test_fish_{fish_idx}"
    os.makedirs(fish_cache_dir, exist_ok=True)

    print(f"  Computing skinning subspace...")
    C = fc.complementary_constraint_matrix(V_centered, T, J, dt=1e-3)
    C2 = fc.lbs_weight_space_constraint(V_centered, C)
    [B, l, Ws] = fc.skinning_subspace(V_centered, T, num_modes, num_clusters, C=C2,
                                     read_cache=False,
                                     cache_dir=fish_cache_dir,
                                     constraint_enforcement="optimal")

    print(f"  Preparing arrays...")
    V_centered = np.ascontiguousarray(V_centered.copy(), dtype=np.float64)
    T = np.ascontiguousarray(T.copy(), dtype=np.int32)
    B = np.ascontiguousarray(B.copy(), dtype=np.float64)
    l = np.ascontiguousarray(l.copy(), dtype=np.int32).reshape(-1, 1)
    J = sp.sparse.csc_matrix(J)

    gc.collect()

    print(f"  Creating simulation...")
    V_sim = np.ascontiguousarray(V_centered.copy(), dtype=np.float64)
    T_sim = np.ascontiguousarray(T.copy(), dtype=np.int32)
    B_sim = np.ascontiguousarray(B.copy(), dtype=np.float64)
    l_sim = np.ascontiguousarray(l.flatten().copy(), dtype=np.int32)
    J_sim = J.copy()

    print(f"  Calling fc.fast_cd_sim...")
    sim = fc.fast_cd_sim(V_sim, T_sim, B_sim, l_sim, J_sim,
                        mu=mu, rho=rho, h=1e-2,
                        cache_dir=fish_cache_dir,
                        read_cache=False,
                        write_cache=False)
    print(f"  ✓ Simulation {fish_idx + 1} created successfully")

    fishes.append({
        'sim': sim,
        'V': V_centered,
        'T': T,
    })

    gc.collect()

print("\n" + "=" * 60)
print("SUCCESS: Both fishes created!")
print("=" * 60)
