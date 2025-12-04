# Guide: Adding Secondary Motion to External Models

This guide explains how to add Fast Cody secondary motion to a model outside this repository.

## Overview

To add secondary motion to an external model, you need:
1. **A tetrahedral mesh** (.msh format)
2. **The Fast Cody library** (fast_cd_pyb and fast_cody Python packages)
3. **Python dependencies**
4. **A simple Python script** to compute and run the simulation

## Step 1: Convert Your Model to MSH Format

Your model needs to be a **tetrahedral mesh** (.msh file). If you have an OBJ file, you need to:

1. **Use TetWild or fTetWild** to convert OBJ → MSH
   - Download from: https://github.com/wildmeshing/fTetWild
   - Or use the conversion pipeline in this repo: `src/fast_cody/apps/convert_to_msh.py`

2. **Basic conversion command:**
   ```bash
   ./FloatTetwild_bin -i your_model.obj -o your_model.msh
   ```

## Step 2: Required Files to Transfer

### Option A: Minimal Transfer (Recommended)
Copy these files/folders to your external project:

```
your_project/
├── fast_cody_minimal/          # Copy these files
│   ├── src/fast_cody/          # Entire fast_cody Python package
│   │   ├── __init__.py
│   │   ├── skinning_subspace.py
│   │   ├── laplacian_eigenmodes.py
│   │   ├── fast_cd_sim.py
│   │   ├── lbs_jacobian.py
│   │   ├── complementary_constraint_matrix.py
│   │   ├── lbs_weight_space_constraint.py
│   │   ├── laplacian.py
│   │   ├── eigs.py
│   │   ├── orthonormalize.py
│   │   ├── skinning_clusters.py
│   │   ├── project_out_subspace.py
│   │   ├── umfpack_lu_solve.py
│   │   └── ... (all other .py files in src/fast_cody/)
│   └── minimal_secondary_motion.py  # Your script (see below)
```

### Option B: Full Installation
Install Fast Cody as a package in your environment:

1. **Clone the repository:**
   ```bash
   git clone --recursive https://github.com/otmanon/fast_cd_pyb
   cd fast_cd_pyb
   ```

2. **Install dependencies:**
   ```bash
   pip install numpy scipy cvxopt scikit-learn
   # Note: libigl is included as a submodule
   ```

3. **Build and install:**
   ```bash
   python setup.py install
   ```

## Step 3: Minimal Python Script

Create a file `minimal_secondary_motion.py`:

```python
import os
import numpy as np
import sys

# Add fast_cody to path (adjust path as needed)
sys.path.insert(0, '/path/to/fast_cody/src')

import fast_cd_pyb as fcd
import fast_cody as fc

def add_secondary_motion(msh_file, output_dir="./cache"):
    """
    Compute secondary motion for a mesh.

    Parameters:
    -----------
    msh_file : str
        Path to .msh tetrahedral mesh file
    output_dir : str
        Directory to save cache files (B.npy, W.npy, l.npy)

    Returns:
    --------
    dict with keys: 'V', 'T', 'B', 'Ws', 'l', 'J', 'sim'
        All necessary data for running secondary motion simulation
    """
    # Load mesh
    [V, F, T] = fcd.readMSH(msh_file)
    print(f"Loaded mesh: {V.shape[0]} vertices, {T.shape[0]} tets")

    # Create cache directory
    os.makedirs(output_dir, exist_ok=True)

    # Scale and center geometry
    [V_centered, so, to] = fcd.scale_and_center_geometry(V, 1, np.array([[0, 0, 0.]]))
    print(f"Centered mesh")

    # Create primary handle (affine transform - all vertices weighted equally)
    Wp = np.ones((V_centered.shape[0], 1))
    J = fc.lbs_jacobian(V_centered, Wp)
    print(f"Created primary handle")

    # Compute constraint matrices
    C = fc.complementary_constraint_matrix(V_centered, T, J, dt=1e-3)
    C2 = fc.lbs_weight_space_constraint(V_centered, C)
    print(f"Computed constraint matrices")

    # Compute secondary motion subspace
    num_modes = 16
    num_clusters = 100
    constraint_enforcement = "optimal"

    print(f"Computing skinning subspace (this may take a while)...")
    [B, l, Ws] = fc.skinning_subspace(
        V_centered, T,
        num_modes=num_modes,
        num_clusters=num_clusters,
        C=C2,
        read_cache=False,
        cache_dir=output_dir,
        constraint_enforcement=constraint_enforcement
    )
    print(f"Computed subspace: B shape={B.shape}, Ws shape={Ws.shape}")

    # Create simulation
    mu = 1e4  # Lame parameter (stiffness)
    rho = 1e3  # Density
    h = 1e-2   # Timestep

    print(f"Creating simulation...")
    sim = fc.fast_cd_sim(
        V_centered, T, B, l, J,
        mu=mu, rho=rho, h=h,
        cache_dir=output_dir,
        read_cache=False
    )
    print(f"Simulation created")

    # Initialize simulation state
    z0 = np.zeros((num_modes * 12, 1))  # Secondary motion coefficients
    T0 = np.identity(4).astype(dtype=np.float32, order="F")
    p0 = T0[0:3, :].reshape((12, 1))  # Primary handle transform
    st = fc.fast_cd_state(z0, p0)

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

def step_simulation(data, primary_transform):
    """
    Step the simulation forward one frame.

    Parameters:
    -----------
    data : dict
        Output from add_secondary_motion()
    primary_transform : (4, 4) numpy array
        Primary handle transform matrix (affine transform)

    Returns:
    --------
    z : (num_modes*12, 1) numpy array
        Secondary motion coefficients
    """
    # Extract primary transform parameters
    p = np.ascontiguousarray(primary_transform[0:3, :].reshape((12, 1)), dtype=np.float64)

    # Step simulation
    z = data['sim'].step(p, data['st'])

    # Update state
    data['st'].update(z, p)

    return z

# Example usage
if __name__ == "__main__":
    # Path to your MSH file
    msh_file = "your_model.msh"

    # Compute secondary motion
    data = add_secondary_motion(msh_file, output_dir="./cache")

    # Example: Apply a transform and get secondary motion
    T = np.identity(4)
    T[0:3, 3] = [1.0, 0.0, 0.0]  # Translate
    z = step_simulation(data, T)

    print(f"Secondary motion computed! z shape: {z.shape}")
```

## Step 4: Required Dependencies

### Python Packages:
```bash
pip install numpy scipy cvxopt scikit-learn
```

### C++ Dependencies (if building from source):
- **CMake** (>= 3.1.0)
- **Eigen3**
- **libigl** (included as submodule)
- **pybind11** (included as submodule)
- **UMFPACK** (from SuiteSparse, for sparse linear algebra)
- **OpenGL** (for viewer, optional if you only need simulation)

### Note on fast_cd_pyb:
The `fast_cd_pyb` module is a compiled C++ extension. You have two options:

1. **Use the pre-built package** (if available for your platform)
2. **Build from source** using `setup.py` (requires C++ compiler and dependencies)

## Step 5: Integration into Your Pipeline

### Basic Integration Pattern:

```python
# 1. Load your mesh
[V, T] = load_your_mesh()  # Your mesh loading code

# 2. Compute secondary motion (one-time setup)
secondary_motion_data = add_secondary_motion("your_model.msh")

# 3. In your animation loop:
for frame in animation_loop:
    # Get your primary transform (from animation, user input, etc.)
    primary_transform = get_primary_transform()  # Your code

    # Step simulation to get secondary motion
    z = step_simulation(secondary_motion_data, primary_transform)

    # Apply deformation to your mesh
    # The secondary motion is encoded in 'z' coefficients
    # You can use B @ z to get vertex displacements
    displacements = secondary_motion_data['B'] @ z
    deformed_vertices = secondary_motion_data['V'] + displacements.reshape(-1, 3)

    # Render deformed_vertices
    render_mesh(deformed_vertices)
```

## Key Files Summary

### Essential Python Files (from `src/fast_cody/`):
- `skinning_subspace.py` - Main function to compute secondary motion
- `fast_cd_sim.py` - Simulation object
- `laplacian_eigenmodes.py` - Eigenmode computation
- `lbs_jacobian.py` - Linear blend skinning jacobian
- `complementary_constraint_matrix.py` - Constraint matrices
- All other supporting files in `src/fast_cody/`

### Compiled Extension:
- `fast_cd_pyb` - C++ extension (must be built or installed)

### Optional (for visualization):
- `src/fast_cody/viewers/` - Viewer code
- `src/shaders/` - GLSL shaders for rendering

## Cache Files

After first computation, these files are saved in `cache_dir`:
- `B.npy` - Subspace matrix (eigenmodes)
- `W.npy` - Skinning weights
- `l.npy` - Cluster indices
- Various `.DMAT` files - Precomputed matrices for simulation

You can set `read_cache=True` on subsequent runs to skip recomputation.

## Troubleshooting

1. **"Module not found: fast_cd_pyb"**
   - You need to build/install the C++ extension
   - Run `python setup.py install` from the fast_cd_pyb repo

2. **"Singular matrix" errors**
   - Your mesh may have degenerate tets
   - Try using `constraint_enforcement="project"` instead of `"optimal"`
   - Check mesh quality with TetWild

3. **Performance issues**
   - Reduce `num_modes` (default 16) for faster computation
   - Reduce `num_clusters` (default 100) for faster clustering
   - Use `read_cache=True` after first computation

## Minimal Example (No Viewer)

If you only need the simulation (no visualization), you can use just:

```python
import fast_cd_pyb as fcd
import fast_cody as fc
import numpy as np

# Load mesh
[V, F, T] = fcd.readMSH("model.msh")
[V, so, to] = fcd.scale_and_center_geometry(V, 1, np.array([[0, 0, 0.]]))

# Setup
Wp = np.ones((V.shape[0], 1))
J = fc.lbs_jacobian(V, Wp)
C = fc.complementary_constraint_matrix(V, T, J, dt=1e-3)
C2 = fc.lbs_weight_space_constraint(V, C)

# Compute
[B, l, Ws] = fc.skinning_subspace(V, T, 16, 100, C=C2, cache_dir="./cache")

# Create sim
sim = fc.fast_cd_sim(V, T, B, l, J, mu=1e4, rho=1e3, h=1e-2, cache_dir="./cache")

# Use
z0 = np.zeros((16*12, 1))
T0 = np.identity(4).astype(np.float32, order="F")
p0 = T0[0:3, :].reshape((12, 1))
st = fc.fast_cd_state(z0, p0)

# Step
p = np.ascontiguousarray(T0[0:3, :].reshape((12, 1)), dtype=np.float64)
z = sim.step(p, st)
st.update(z, p)
```

This gives you everything needed for secondary motion without the viewer!
