#!/usr/bin/env python3
"""
Test script to isolate the segfault issue by creating fishes one at a time
"""
import sys
import os
import numpy as np

# Add the source directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import fast_cody as fc
import fast_cd_pyb as fcd

# Test loading and creating simulation for each mesh individually
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

print("=" * 80)
print("TEST 1: Default fish mesh")
print("=" * 80)

default_fish_msh = fc.get_data("./cd_fish.msh")
[V1, F1, T1] = fcd.readMSH(default_fish_msh)
print(f"Loaded default fish: V shape={V1.shape}, T shape={T1.shape}")

# Center geometry
[V1_centered, so1, to1] = fcd.scale_and_center_geometry(V1, 1, np.array([[0, 0, 0.]]))

# Create handle
Wp1 = np.ones((V1_centered.shape[0], 1))
J1 = fc.lbs_jacobian(V1_centered, Wp1)

print("Computing skinning subspace for default fish...")
C1 = fc.complementary_constraint_matrix(V1_centered, T1, J1, dt=1e-3)
C2_1 = fc.lbs_weight_space_constraint(V1_centered, C1)
[B1, l1, Ws1] = fc.skinning_subspace(V1_centered, T1, 16, 100, C=C2_1,
                                     read_cache=False,
                                     cache_dir="./cache/test_fish_0",
                                     constraint_enforcement="optimal")
print(f"Skinning subspace computed: B shape={B1.shape}, l shape={l1.shape}")

print("Creating simulation for default fish...")
import scipy as sp
J1_sparse = sp.sparse.csc_matrix(J1)
sim1 = fc.fast_cd_sim(V1_centered, T1, B1, l1.flatten(), J1_sparse,
                     mu=1e4, rho=1e3, h=1e-2,
                     cache_dir="./cache/test_fish_0",
                     read_cache=False,
                     write_cache=False)
print("✓ Simulation 1 created successfully\n")

# Clean up
del sim1, B1, l1, Ws1, J1_sparse, J1, C1, C2_1
import gc
gc.collect()

print("=" * 80)
print("TEST 2: Same fish mesh (but separate instance)")
print("=" * 80)

# Use the same mesh file but create a completely separate instance
new_model_msh = fc.get_data("./cd_fish.msh")

[V2, F2, T2] = fcd.readMSH(new_model_msh)
print(f"Loaded custom fish: V shape={V2.shape}, T shape={T2.shape}")

# Center geometry
[V2_centered, so2, to2] = fcd.scale_and_center_geometry(V2, 1, np.array([[0, 0, 0.]]))

# Create handle
Wp2 = np.ones((V2_centered.shape[0], 1))
J2 = fc.lbs_jacobian(V2_centered, Wp2)

print("Computing skinning subspace for custom fish...")
C2 = fc.complementary_constraint_matrix(V2_centered, T2, J2, dt=1e-3)
C2_2 = fc.lbs_weight_space_constraint(V2_centered, C2)
[B2, l2, Ws2] = fc.skinning_subspace(V2_centered, T2, 16, 100, C=C2_2,
                                     read_cache=False,
                                     cache_dir="./cache/test_fish_1",
                                     constraint_enforcement="optimal")
print(f"Skinning subspace computed: B shape={B2.shape}, l shape={l2.shape}")

print("Creating simulation for custom fish...")
J2_sparse = sp.sparse.csc_matrix(J2)
sim2 = fc.fast_cd_sim(V2_centered, T2, B2, l2.flatten(), J2_sparse,
                     mu=1e4, rho=1e3, h=1e-2,
                     cache_dir="./cache/test_fish_1",
                     read_cache=False,
                     write_cache=False)
print("✓ Simulation 2 created successfully\n")

print("=" * 80)
print("SUCCESS: Both simulations created without segfault!")
print("=" * 80)
