#!/usr/bin/env python3
"""
Minimal example: Adding secondary motion to an external model

This script demonstrates the minimal code needed to add Fast Cody secondary motion
to a model outside this repository.

Requirements:
- fast_cd_pyb and fast_cody packages installed
- A .msh tetrahedral mesh file
- Python dependencies: numpy, scipy, cvxopt, scikit-learn
"""

import os
import numpy as np
import sys

# Import Fast Cody (adjust import path as needed)
# Option 1: If installed as package
# import fast_cd_pyb as fcd
# import fast_cody as fc

# Option 2: If using from source (uncomment and adjust path)
# sys.path.insert(0, '/path/to/fast_cody/src')
# import fast_cd_pyb as fcd
# import fast_cody as fc


def compute_secondary_motion(msh_file, cache_dir="./cache", num_modes=16, num_clusters=100):
    """
    Compute secondary motion for a tetrahedral mesh.

    This is the core function - copy this to your project!

    Parameters:
    -----------
    msh_file : str
        Path to .msh tetrahedral mesh file
    cache_dir : str
        Directory to save/load cache files
    num_modes : int
        Number of eigenmodes (default: 16, higher = more detail but slower)
    num_clusters : int
        Number of clusters for skinning (default: 100)

    Returns:
    --------
    dict containing:
        - 'V': (n, 3) vertex positions (centered)
        - 'T': (t, 4) tet indices
        - 'B': (3*n, num_modes*12) subspace matrix
        - 'Ws': (n, num_modes) skinning weights
        - 'l': (t,) cluster indices
        - 'J': (12, 3*n) LBS jacobian
        - 'sim': fast_cd_sim object
        - 'st': fast_cd_state object
        - 'num_modes': number of modes
    """
    import fast_cd_pyb as fcd
    import fast_cody as fc

    print(f"Loading mesh: {msh_file}")
    [V, F, T] = fcd.readMSH(msh_file)
    print(f"  Vertices: {V.shape[0]}, Tets: {T.shape[0]}")

    # Scale and center geometry
    [V_centered, so, to] = fcd.scale_and_center_geometry(V, 1, np.array([[0, 0, 0.]]))
    print(f"  Centered and scaled")

    # Create primary handle (affine transform)
    Wp = np.ones((V_centered.shape[0], 1))
    J = fc.lbs_jacobian(V_centered, Wp)
    print(f"  Created primary handle")

    # Compute constraint matrices
    C = fc.complementary_constraint_matrix(V_centered, T, J, dt=1e-3)
    C2 = fc.lbs_weight_space_constraint(V_centered, C)
    print(f"  Computed constraints")

    # Create cache directory
    os.makedirs(cache_dir, exist_ok=True)

    # Compute secondary motion subspace
    print(f"  Computing skinning subspace (this may take a few minutes)...")
    [B, l, Ws] = fc.skinning_subspace(
        V_centered, T,
        num_modes=num_modes,
        num_clusters=num_clusters,
        C=C2,
        read_cache=False,  # Set to True after first run
        cache_dir=cache_dir,
        constraint_enforcement="optimal"  # or "project" if singular matrix errors
    )
    print(f"  Subspace computed: B={B.shape}, Ws={Ws.shape}")

    # Create simulation
    mu = 1e4   # Stiffness (Lame parameter)
    rho = 1e3  # Density
    h = 1e-2   # Timestep

    print(f"  Creating simulation...")
    sim = fc.fast_cd_sim(
        V_centered, T, B, l, J,
        mu=mu, rho=rho, h=h,
        cache_dir=cache_dir,
        read_cache=False
    )

    # Initialize state
    z0 = np.zeros((num_modes * 12, 1))
    T0 = np.identity(4).astype(dtype=np.float32, order="F")
    p0 = T0[0:3, :].reshape((12, 1))
    st = fc.fast_cd_state(z0, p0)

    print(f"  Secondary motion setup complete!")

    return {
        'V': V_centered,
        'T': T,
        'B': B,
        'Ws': Ws,
        'l': l,
        'J': J,
        'sim': sim,
        'st': st,
        'num_modes': num_modes
    }


def apply_transform(data, transform_matrix):
    """
    Apply a transform and get secondary motion.

    Parameters:
    -----------
    data : dict
        Output from compute_secondary_motion()
    transform_matrix : (4, 4) numpy array
        Affine transform matrix (primary handle)

    Returns:
    --------
    z : (num_modes*12, 1) numpy array
        Secondary motion coefficients
    """
    # Extract transform parameters (12 values: 3x4 matrix flattened)
    p = np.ascontiguousarray(transform_matrix[0:3, :].reshape((12, 1)), dtype=np.float64)

    # Step simulation
    z = data['sim'].step(p, data['st'])

    # Update state
    data['st'].update(z, p)

    return z


def get_deformed_vertices(data, transform_matrix):
    """
    Get deformed vertex positions with secondary motion.

    Parameters:
    -----------
    data : dict
        Output from compute_secondary_motion()
    transform_matrix : (4, 4) numpy array
        Affine transform matrix

    Returns:
    --------
    V_deformed : (n, 3) numpy array
        Deformed vertex positions
    """
    # Get secondary motion
    z = apply_transform(data, transform_matrix)

    # Apply secondary motion to vertices
    # B is (3*n, num_modes*12), z is (num_modes*12, 1)
    displacements = (data['B'] @ z).reshape(-1, 3)

    # Apply primary transform
    V_homogeneous = np.hstack([data['V'], np.ones((data['V'].shape[0], 1))])
    V_transformed = (transform_matrix @ V_homogeneous.T).T[:, :3]

    # Add secondary motion
    V_deformed = V_transformed + displacements

    return V_deformed


# Example usage
if __name__ == "__main__":
    # Path to your MSH file
    msh_file = "your_model.msh"  # Change this!

    if not os.path.exists(msh_file):
        print(f"ERROR: Mesh file not found: {msh_file}")
        print("Please provide a .msh tetrahedral mesh file.")
        sys.exit(1)

    # Compute secondary motion (one-time setup, can be cached)
    print("=" * 60)
    print("Computing Secondary Motion")
    print("=" * 60)
    data = compute_secondary_motion(msh_file, cache_dir="./cache")

    # Example: Apply some transforms
    print("\n" + "=" * 60)
    print("Testing Secondary Motion")
    print("=" * 60)

    # Example 1: Identity (no transform)
    T1 = np.identity(4)
    z1 = apply_transform(data, T1)
    print(f"Identity transform: z norm = {np.linalg.norm(z1):.6f}")

    # Example 2: Translation
    T2 = np.identity(4)
    T2[0:3, 3] = [0.5, 0.0, 0.0]  # Translate in X
    z2 = apply_transform(data, T2)
    print(f"Translation: z norm = {np.linalg.norm(z2):.6f}")

    # Example 3: Rotation
    angle = np.pi / 4
    T3 = np.identity(4)
    T3[0:3, 0:3] = np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1]
    ])
    z3 = apply_transform(data, T3)
    print(f"Rotation: z norm = {np.linalg.norm(z3):.6f}")

    # Get deformed vertices
    V_deformed = get_deformed_vertices(data, T2)
    print(f"\nDeformed vertices shape: {V_deformed.shape}")
    print(f"Max displacement: {np.max(np.linalg.norm(V_deformed - data['V'], axis=1)):.6f}")

    print("\n" + "=" * 60)
    print("Success! Secondary motion is working.")
    print("=" * 60)
    print("\nTo use in your project:")
    print("1. Copy compute_secondary_motion() function")
    print("2. Copy apply_transform() or get_deformed_vertices() functions")
    print("3. Call compute_secondary_motion() once (cache the result)")
    print("4. Call apply_transform() each frame with your animation transforms")
