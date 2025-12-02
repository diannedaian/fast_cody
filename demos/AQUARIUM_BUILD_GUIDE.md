# Interactive Aquarium Build Guide

## Overview
This guide explains how to build an interactive aquarium with 10 fish, each with its own affine handle, secondary motion, and preset swimming animations using the fast_cody repository.

## Architecture Overview

The aquarium combines two key approaches:
1. **Multi-fish affine handles** (`interactive_cd_affine_handle_multi_fish`) - for independent control of each fish
2. **Rig animations** (`interactive_cd_rig_anim`) - for swimming motion

Each fish will have:
- An affine handle (4x4 transform matrix) for position/orientation control
- A rig animation (swimming motion) that loops continuously
- Secondary motion effects from the fast CD simulation

## Key Components

### 1. Fish Model and Assets
- **Mesh**: `data/cd_fish/cd_fish.msh` (tetrahedral mesh)
- **Texture**: `data/cd_fish/cd_fish_tex.png` (texture image)
- **Texture OBJ**: `data/cd_fish/cd_fish_tex.obj` (surface mesh with UV coordinates)
- **Rig**: `data/cd_fish/rigs/skeleton_rig/skeleton_rig.json` (bone structure)
- **Animation**: `data/cd_fish/rigs/skeleton_rig/anim/swim.json` (swimming animation)

### 2. Core Functions to Reuse

#### From `interactive_cd_affine_handle_multi_fish.py`:
- Multi-fish setup and initialization
- Separate simulation objects for each fish
- Viewer setup with multiple meshes
- Guizmo controls for interactive manipulation
- Floor and caustics rendering

#### From `interactive_cd_rig_anim.py`:
- Rig loading: `fcd.read_rig_from_json(rig_file)`
- Animation loading: `fcd.read_rig_anim_from_json(anim_file)`
- Weight diffusion: `fcd.diffuse_weights(V, T, Wpsurface, bI, dt=10000)`
- World-to-relative transform: `fcd.world2rel(P, P0)`
- Animation playback loop

## Implementation Strategy

### Step 1: Load Fish Data
```python
import fast_cody as fcd
import igl
import numpy as np

# Load mesh
msh_file = fcd.get_data("./cd_fish.msh")
[V, F, T] = fcd.read_msh(msh_file)

# Load rig
rig_file = fcd.get_data("./cd_fish_rig.json")
[Vpsurf, Fpsurf, Wpsurface, P0, lengths, pI] = fcd.read_rig_from_json(rig_file)

# Compute primary weights (Wp) for affine handle
aI = np.arange(V.shape[0])
[D2, bI, CP] = igl.point_mesh_squared_distance(Vpsurf, V, aI)
Wp = fcd.diffuse_weights(V, T, Wpsurface, bI, dt=10000)

# Load animation
anim_file = fcd.get_data("./cd_fish_rig_anim__swim.json")
P = fcd.read_rig_anim_from_json(anim_file)
```

### Step 2: Setup Multiple Fish

For each of the 10 fish:
1. **Create independent simulation objects** (like `interactive_cd_affine_handle_multi_fish`)
2. **Initialize with different positions** (spread them out in 3D space)
3. **Each fish gets its own:**
   - Affine handle transform (T0) for position/orientation
   - Animation state (current frame in the swim animation)
   - Simulation state (z, p) for secondary motion

### Step 3: Combine Affine Handle + Animation

The key insight: The affine handle controls the **global transform** (where the fish is in the tank), while the rig animation controls the **local swimming motion**.

```python
# For each fish in the pre_draw_callback:
# 1. Update animation frame (loop through swim animation)
current_frame = (step + fish_offset) % num_frames
p_anim = Prel[:, current_frame]  # Animation transform (relative to P0)

# 2. Get affine handle transform (global position/orientation)
T0_affine = fish['T0']  # 4x4 matrix from guizmo/user input

# 3. Combine: Apply affine handle to the animated rig
# The animation P is relative to P0, so we need to:
# - Start with P0 (initial rig configuration)
# - Apply animation: P0 + animation_delta
# - Apply affine handle transform to the whole thing

# Convert T0 to rig space and combine with animation
p_combined = combine_affine_and_animation(T0_affine, P0, p_anim)

# 4. Step simulation with combined transform
z = fish['sim'].step(p_combined, fish['st'])
fish['st'].update(z, p_combined)

# 5. Update viewer
viewer_base.set_bone_transforms(p_combined, z, mesh_id)
```

### Step 4: Animation Combination Logic

The animation `P` is stored as `(bones x timesteps x 3 x 4)` in world space, but needs to be converted to relative space and combined with the affine handle:

```python
# From interactive_cd_rig_anim.py:
[V, so, to] = fcd.scale_and_center_geometry(V, 1, np.array([[0, 0, 0.]]))
P0 = P0 * so
P0[:, :, 3] = P0[:, :, 3] - to

P = P * so
P[:, :, :, 3] = P[:, :, :, 3] - to
Prel = fcd.world2rel(P, P0)  # Convert to relative transforms

# Reshape for simulation: (bones * 12, frames)
d = V.shape[1]  # 3
k = Wp.shape[1]  # number of bones
frames = P.shape[0]
Prel = np.transpose(Prel, [3, 1, 2, 0])  # (12, bones, frames)
Prel = Prel.reshape(((d + 1) * d * k, frames), order='F')  # (bones*12, frames)
```

For combining with affine handle:
- The affine handle T0 is a single 4x4 transform
- The animation Prel has multiple bones (k bones)
- The first bone (index 0) should be multiplied by the affine handle
- Other bones follow the animation

### Step 5: Simple Preset Animation Paths

For automatic movement, update each fish's T0 over time:

```python
# In pre_draw_callback, for each fish:
if auto_animate:
    # Simple circular or figure-8 path
    t = step * 0.01  # Time parameter
    radius = 2.0
    speed = 0.5

    # Circular path
    fish['T0'][0, 3] = radius * np.cos(t * speed + fish_idx * 0.6)  # X
    fish['T0'][1, 3] = 0.0  # Y (height)
    fish['T0'][2, 3] = radius * np.sin(t * speed + fish_idx * 0.6)  # Z

    # Orient fish in direction of movement
    # (compute rotation based on velocity direction)
```

## Code Structure

### Main Function Signature
```python
def interactive_aquarium(
    num_fishes=10,
    msh_file=None,
    texture_png=None,
    texture_obj=None,
    rig_file=None,
    anim_file=None,
    fish_positions=None,
    auto_animate=True,
    enable_caustics=True,
    mu=1e4,
    rho=1e3,
    num_modes=16,
    num_clusters=100
):
```

### Fish Data Structure
Each fish is stored as a dictionary:
```python
fish = {
    'V': V_centered,           # Centered vertices
    'T': T,                    # Tetrahedra
    'F': F,                    # Surface faces
    'Wp': Wp,                  # Primary weights (affine handle)
    'Ws': Ws,                  # Secondary weights
    'B': B,                    # Skinning subspace basis
    'l': l,                    # Cluster indices
    'J': J,                    # LBS Jacobian
    'sim': sim,                # Fast CD simulation object
    'st': st,                  # Simulation state
    'T0': T0,                  # Affine handle transform (4x4)
    'P0': P0,                  # Initial rig configuration
    'Prel': Prel,              # Relative animation transforms
    'anim_frame': 0,           # Current animation frame
    'anim_offset': offset,     # Frame offset for variety
    'position_offset': pos,    # Initial position
    'so': so,                  # Scale factor
    'to': to,                  # Translation offset
    'texture_png': texture_png,
    'texture_obj': texture_obj
}
```

### Pre-draw Callback Structure
```python
def pre_draw_callback():
    nonlocal step

    for fish_idx, fish in enumerate(fishes):
        # 1. Update animation frame (with offset for variety)
        anim_frame = (step + fish['anim_offset']) % num_frames
        p_anim = fish['Prel'][:, anim_frame]

        # 2. Get/update affine handle (from guizmo or auto-animation)
        if auto_animate:
            update_fish_path(fish, step, fish_idx)

        T0 = fish['T0']
        p_affine = T0[0:3, :].reshape((12, 1))

        # 3. Combine affine handle with animation
        # For single-bone rig, affine handle replaces first bone
        # For multi-bone rig, affine handle multiplies first bone
        p_combined = combine_transforms(p_affine, p_anim, fish['P0'])

        # 4. Step simulation
        z = fish['sim'].step(p_combined, fish['st'])
        fish['st'].update(z, p_combined)

        # 5. Update viewer
        mesh_id = mesh_ids[fish_idx]
        viewer_base.set_bone_transforms(p_combined, z, mesh_id)
        viewer_base.updateGL(mesh_id)

    step += 1
```

## Key Implementation Details

### 1. Transform Combination
The affine handle (T0) and rig animation (P) need to be combined correctly:
- If using a single-bone rig (affine handle only), T0 directly becomes p
- If using multi-bone rig, T0 should multiply the root bone's transform
- Animation P is relative to P0, so: `P_world = P0 + Prel`

### 2. Animation Offsets
Give each fish a different animation offset so they don't all swim in sync:
```python
anim_offsets = [i * (num_frames // num_fishes) for i in range(num_fishes)]
```

### 3. Position Initialization
Spread fish out in 3D space:
```python
fish_positions = []
for i in range(num_fishes):
    angle = 2 * np.pi * i / num_fishes
    radius = 1.5
    fish_positions.append(np.array([
        radius * np.cos(angle),
        0.0,  # Height
        radius * np.sin(angle)
    ]))
```

### 4. Secondary Motion Scaling
Control the intensity of secondary motion:
```python
secondary_motion_scale = 1.0  # Reduce for subtler effects
z_scaled = z * secondary_motion_scale
```

## Example Usage

```python
import fast_cody as fcd

# Simple call with defaults
fcd.apps.interactive_aquarium(
    num_fishes=10,
    auto_animate=True,
    enable_caustics=True
)

# Custom setup
fcd.apps.interactive_aquarium(
    num_fishes=10,
    msh_file="data/cd_fish/cd_fish.msh",
    texture_png="data/cd_fish/cd_fish_tex.png",
    texture_obj="data/cd_fish/cd_fish_tex.obj",
    rig_file="data/cd_fish/rigs/skeleton_rig/skeleton_rig.json",
    anim_file="data/cd_fish/rigs/skeleton_rig/anim/swim.json",
    auto_animate=True,
    enable_caustics=True
)
```

## Files to Reference

1. **`demos/affine_handle_with_floor_multi_fish.py`** - Multi-fish setup
2. **`demos/rig_anim.py`** - Rig animation loading and playback
3. **`src/fast_cody/apps/interactive_cd_affine_handle_multi_fish.py`** - Full multi-fish implementation
4. **`src/fast_cody/apps/interactive_cd_rig_anim.py`** - Rig animation implementation

## Challenges and Solutions

### Challenge 1: Combining Affine Handle with Multi-bone Animation
**Solution**: The affine handle should control the root bone. For a multi-bone rig:
- Extract root bone transform from Prel
- Multiply by affine handle T0
- Keep other bones as-is from animation

### Challenge 2: Animation Timing
**Solution**: Use frame offsets and different animation speeds per fish:
```python
fish['anim_speed'] = 0.8 + random.uniform(-0.2, 0.2)  # Vary speed
anim_frame = int((step * fish['anim_speed'] + fish['anim_offset']) % num_frames)
```

### Challenge 3: Performance with 10 Fish
**Solution**:
- Use `read_cache=True` after first fish
- Reuse same mesh/rig data for all fish
- Consider reducing `num_modes` if needed

## Next Steps

1. Create `interactive_aquarium.py` in `src/fast_cody/apps/`
2. Implement transform combination logic
3. Add simple path-following for auto-animation
4. Test with 2-3 fish first, then scale to 10
5. Add controls to toggle auto-animation on/off

## Quick Reference

### Essential File Paths
- Mesh: `data/cd_fish/cd_fish.msh` or `fcd.get_data("./cd_fish.msh")`
- Texture PNG: `data/cd_fish/cd_fish_tex.png` or `fcd.get_data("./cd_fish_tex.png")`
- Texture OBJ: `data/cd_fish/cd_fish_tex.obj` or `fcd.get_data("./cd_fish_tex.obj")`
- Rig: `data/cd_fish/rigs/skeleton_rig/skeleton_rig.json`
- Animation: `data/cd_fish/rigs/skeleton_rig/anim/swim.json`

### Key API Functions
```python
# Loading
[V, F, T] = fcd.read_msh(msh_file)
[Vpsurf, Fpsurf, Wpsurface, P0, lengths, pI] = fcd.read_rig_from_json(rig_file)
P = fcd.read_rig_anim_from_json(anim_file)

# Processing
Wp = fcd.diffuse_weights(V, T, Wpsurface, bI, dt=10000)
[V, so, to] = fcd.scale_and_center_geometry(V, 1, np.array([[0, 0, 0.]]))
Prel = fcd.world2rel(P, P0)

# Simulation
J = fcd.lbs_jacobian(V, Wp)
[B, l, Ws] = fcd.skinning_subspace(V, T, num_modes, num_clusters, ...)
sim = fcd.fast_cd_sim(V, T, B, l, J, mu=mu, rho=rho, h=1e-2, ...)
st = fcd.fast_cd_state(z0, p0)
z = sim.step(p, st)
```

### Transform Format
- **T0 (affine handle)**: 4x4 matrix, stored as `T0[0:3, :]` for 3x4 part
- **p (rig parameters)**: Flattened to (bones*12, 1) vector
  - Format: [R11, R21, R31, R12, R22, R32, R13, R23, R33, tx, ty, tz] per bone
- **Prel (animation)**: (frames, bones, 3, 4) → reshaped to (bones*12, frames)

### Implementation Checklist
- [ ] Load mesh, rig, and animation data
- [ ] Create 10 fish instances with separate simulations
- [ ] Initialize fish positions in 3D space
- [ ] Set up animation offsets for variety
- [ ] Implement transform combination (affine handle + animation)
- [ ] Create pre-draw callback to update all fish
- [ ] Add auto-animation paths (optional)
- [ ] Set up viewer with multiple meshes
- [ ] Add floor and caustics (optional)
- [ ] Test and optimize performance
