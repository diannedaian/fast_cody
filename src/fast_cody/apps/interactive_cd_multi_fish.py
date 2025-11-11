import os
import numpy as np
import scipy as sp
import igl
import json
from os.path import basename, splitext

import fast_cd_pyb as fcd
import fast_cody as fc


def interactive_cd_multi_fish(msh_files=None, Vs=None, Ts=None, Ws_list=None, l_list=None,
                              mu=1e4, rho=1e3, num_modes=16, num_clusters=100,
                              constraint_enforcement="optimal",
                              cache_dir=None, results_dir=None, read_cache=False,
                              texture_png_list=None, texture_obj_list=None,
                              num_fishes=2, fish_positions=None):
    """
    Runs an interactive fast CD simulation with multiple fishes, where the user can control
    each fish independently using an affine handle with a Guizmo.

    Parameters
    ----------
    msh_files : list of str or str
        List of paths to Tet mesh .msh files, or a single path (will be duplicated).
        If None, uses default fish mesh.
    Vs : list of (n, 3) float numpy arrays
        List of vertex positions for each fish. If None, expects msh_files to be provided.
    Ts : list of (t, 4) int numpy arrays
        List of tet indices for each fish. If None, expects msh_files to be provided.
    Ws_list : list of (n, m) float numpy arrays
        List of skinning weights for each fish. If None, recomputed on the fly.
    l_list : list of (t, 1) int numpy arrays
        List of per-tet cluster indices for each fish. If None, recomputed on the fly.
    mu : float
        First Lame parameter (default=1e4)
    rho : float
        Density (default=1e3)
    num_modes : int
        Number of skinning modes to compute if Ws_list is None
    num_clusters : int
        Number of skinning clusters to compute if l_list is None
    constraint_enforcement : str
        {"project", "optimal"}. If "optimal", performs the full constrained GEVP.
    cache_dir : str
        Directory where results are stored and where cache is stored.
    read_cache : bool
        Whether to read skinning modes from cache or not (default=False)
    texture_png_list : list of str
        List of paths to texture PNG files for each fish.
    texture_obj_list : list of str
        List of paths to texture OBJ files for each fish.
    num_fishes : int
        Number of fishes to create (default=2)
    fish_positions : list of (3,) float numpy arrays
        Initial positions for each fish. If None, fishes are positioned side by side.

    Examples
    --------
    >>> import fast_cody as fc
    >>> fc.apps.interactive_cd_multi_fish()
    """

    # Handle input: normalize to lists
    if msh_files is None:
        default_msh = fc.get_data("./cd_fish.msh")
        msh_files = [default_msh] * num_fishes
    elif isinstance(msh_files, str):
        msh_files = [msh_files] * num_fishes

    if Vs is None:
        Vs = []
        Ts = []
        for msh_file in msh_files:
            [V, F, T] = fcd.readMSH(msh_file)
            Vs.append(V)
            Ts.append(T)
    elif not isinstance(Vs, list):
        Vs = [Vs] * num_fishes
        Ts = [Ts] * num_fishes

    # Set up textures
    if texture_png_list is None:
        texture_png_list = []
        texture_obj_list = []
        for msh_file in msh_files:
            if msh_file == fc.get_data("./cd_fish.msh"):
                texture_png_list.append(fc.get_data("./cd_fish_tex.png"))
                texture_obj_list.append(fc.get_data("./cd_fish_tex.obj"))
            else:
                texture_png_list.append(None)
                texture_obj_list.append(None)
    elif isinstance(texture_png_list, str):
        texture_png_list = [texture_png_list] * num_fishes
        texture_obj_list = [texture_obj_list] * num_fishes

    if cache_dir is None:
        cache_dir = "./cache/"
    os.makedirs(cache_dir, exist_ok=True)

    # Set up fish positions
    if fish_positions is None:
        fish_positions = []
        for i in range(num_fishes):
            # Position fishes side by side
            offset = (i - (num_fishes - 1) / 2) * 1.5
            fish_positions.append(np.array([offset, 0.0, 0.0]))

    # Process each fish - use separate cache directories to avoid conflicts
    # Create simulations sequentially to avoid conflicts
    fishes = []

    for fish_idx in range(num_fishes):
        print(f"Creating fish {fish_idx + 1}/{num_fishes}...")
        V = Vs[fish_idx].copy()
        T = Ts[fish_idx].copy()

        # Scale and center geometry (compute subspace on centered geometry)
        [V_centered, so, to] = fcd.scale_and_center_geometry(V, 1, np.array([[0, 0, 0.]]))

        # Create primary handle on centered geometry
        Wp = np.ones((V_centered.shape[0], 1))
        J = fc.lbs_jacobian(V_centered, Wp)

        # For identical meshes, use the same cache directory to ensure identical simulation data
        # This guarantees both fishes use exactly the same precomputed matrices
        if fish_idx == 0:
            fish_cache_dir = os.path.join(cache_dir, "fish_0")
        else:
            # Use the same cache as first fish to ensure identical simulation behavior
            fish_cache_dir = os.path.join(cache_dir, "fish_0")

        os.makedirs(fish_cache_dir, exist_ok=True)

        # Compute or use provided skinning weights (on centered geometry)
        if Ws_list is None or l_list is None or fish_idx >= len(Ws_list) or fish_idx >= len(l_list):
            C = fc.complementary_constraint_matrix(V_centered, T, J, dt=1e-3)
            C2 = fc.lbs_weight_space_constraint(V_centered, C)
            [B, l, Ws] = fc.skinning_subspace(V_centered, T, num_modes, num_clusters, C=C2,
                                             read_cache=read_cache,
                                             cache_dir=fish_cache_dir,
                                             constraint_enforcement=constraint_enforcement)
        else:
            Ws = Ws_list[fish_idx]
            l = l_list[fish_idx]
            num_modes = Ws.shape[1]
            num_clusters = l.max() + 1
            # Need to compute B from Ws - this is a simplification
            # In practice, B should be stored/cached with Ws
            C = fc.complementary_constraint_matrix(V_centered, T, J, dt=1e-3)
            C2 = fc.lbs_weight_space_constraint(V_centered, C)
            [B, _, _] = fc.skinning_subspace(V_centered, T, num_modes, num_clusters, C=C2,
                                            read_cache=read_cache,
                                            cache_dir=fish_cache_dir,
                                            constraint_enforcement=constraint_enforcement)

        # Ensure all arrays are contiguous and properly formatted before creating simulation
        # Make deep copies to ensure complete independence from other fishes
        V_centered = np.ascontiguousarray(V_centered.copy(), dtype=np.float64)
        T = np.ascontiguousarray(T.copy(), dtype=np.int32)
        B = np.ascontiguousarray(B.copy(), dtype=np.float64)
        l = np.ascontiguousarray(l.copy(), dtype=np.int32).reshape(-1, 1)
        # J is already a sparse matrix from lbs_jacobian, make a copy to ensure independence
        if isinstance(J, sp.sparse.csc_matrix):
            J = J.copy()
        else:
            J = sp.sparse.csc_matrix(J)

        # Create simulation (on centered geometry) - use separate cache for each fish
        # Also ensure we don't write cache simultaneously to avoid conflicts
        # Force garbage collection before creating simulation to ensure clean state
        import gc
        gc.collect()

        # Create simulation - each fish gets its own simulation object
        # Even for identical meshes, we need separate simulation objects because
        # each simulation holds references to specific array objects
        print(f"  Creating simulation for fish {fish_idx + 1}...")

        # Ensure all data is properly formatted and independent
        # Make sure arrays are contiguous and properly typed
        V_sim = np.ascontiguousarray(V_centered, dtype=np.float64)
        T_sim = np.ascontiguousarray(T, dtype=np.int32)
        B_sim = np.ascontiguousarray(B, dtype=np.float64)
        l_sim = np.ascontiguousarray(l.flatten(), dtype=np.int32)
        J_sim = J.copy() if isinstance(J, sp.sparse.csc_matrix) else sp.sparse.csc_matrix(J)

        try:
            # Create simulation - use read_cache=True for subsequent fishes to avoid recomputation
            # This should help avoid segfaults from recomputing precomputations
            use_read_cache = read_cache or (fish_idx > 0)  # Always try to read cache for fish > 0
            sim = fc.fast_cd_sim(V_sim, T_sim, B_sim, l_sim, J_sim,
                                mu=mu, rho=rho, h=1e-2,
                                cache_dir=fish_cache_dir,
                                read_cache=use_read_cache,
                                write_cache=(fish_idx == 0))  # Only write cache for first fish
            print(f"  Simulation created successfully for fish {fish_idx + 1}")
        except Exception as e:
            print(f"Error creating simulation for fish {fish_idx}: {e}")
            import traceback
            traceback.print_exc()
            raise

        # Initialize T0 with position offset for visualization
        T0 = np.identity(4).astype(dtype=np.float32, order="F")
        T0[0:3, 3] = fish_positions[fish_idx]

        # Store initial T0 for computing relative transforms later
        T0_initial = T0.copy()

        # Initialize state with identity transform (no position offset)
        # This ensures both fishes have identical simulation behavior
        # The position offset will be applied only during rendering
        z0 = np.zeros((num_modes * 12, 1), dtype=np.float64)
        T0_identity = np.identity(4).astype(dtype=np.float32, order="F")
        p0 = T0_identity[0:3, :].reshape((12, 1)).astype(dtype=np.float64)
        st = fc.fast_cd_state(z0, p0)

        # For visualization: use centered vertices (position offset is in transform, not vertices)
        # The shader will apply the transform which includes the position offset
        V_vis = V_centered.copy()

        fishes.append({
            'V': V_vis,  # Visual vertices (centered, position comes from transform)
            'V_centered': V_centered,  # Centered vertices for simulation (keep reference)
            'T': T,  # Keep reference to T
            'F': igl.boundary_facets(T)[0],
            'Wp': Wp,
            'Ws': Ws,
            'B': B,  # Keep reference to B
            'l': l,  # Keep reference to l
            'J': J,  # Keep reference to J
            'sim': sim,  # Keep reference to simulation
            'st': st,  # Keep reference to state
            'T0': T0,
            'T0_initial': T0_initial,  # Store initial T0 for relative transform computation
            'so': so,
            'to': to,
            'position_offset': fish_positions[fish_idx],
            'texture_png': texture_png_list[fish_idx] if fish_idx < len(texture_png_list) else None,
            'texture_obj': texture_obj_list[fish_idx] if fish_idx < len(texture_obj_list) else None,
        })

    # Create viewer with multiple meshes
    vertex_shader_path = fc.get_shader("./vertex_shader_16.glsl")
    fragment_shader_path = fc.get_shader("./fragment_shader.glsl")

    viewer_base = fcd.fast_cd_viewer_custom_shader(vertex_shader_path,
                                                   fragment_shader_path, 16, 16)

    # Add all meshes to viewer
    # First, create all mesh slots
    print("Adding meshes to viewer...")
    mesh_ids = []
    for fish_idx in range(num_fishes):
        mesh_id = viewer_base.add_mesh()
        mesh_ids.append(mesh_id)
        print(f"  Created mesh slot {mesh_id} for fish {fish_idx + 1}")

    # Then, set mesh data for each fish
    for fish_idx, fish in enumerate(fishes):
        mesh_id = mesh_ids[fish_idx]
        print(f"  Setting up mesh {mesh_id} for fish {fish_idx + 1}...")

        vis_texture = (fish['texture_png'] is not None and
                      fish['texture_obj'] is not None)

        if not vis_texture:
            try:
                # Ensure arrays are contiguous
                print(f"      Preparing arrays for mesh {mesh_id}...")
                V_vis = np.ascontiguousarray(fish['V'], dtype=np.float64)
                F_vis = np.ascontiguousarray(fish['F'], dtype=np.int32)
                print(f"      V shape: {V_vis.shape}, F shape: {F_vis.shape}")

                print(f"      Calling set_mesh for mesh {mesh_id}...")
                viewer_base.set_mesh(V_vis, F_vis, mesh_id)
                print(f"      set_mesh completed")

                print(f"      Calling invert_normals for mesh {mesh_id}...")
                viewer_base.invert_normals(True, mesh_id)
                print(f"      invert_normals completed")

                # Different colors for different fishes
                colors = [
                    np.array([144, 210, 236]) / 255.0,  # Blue
                    np.array([236, 144, 144]) / 255.0,  # Red
                    np.array([144, 236, 144]) / 255.0,  # Green
                    np.array([236, 236, 144]) / 255.0,  # Yellow
                ]
                color = colors[fish_idx % len(colors)]
                print(f"      Calling set_color for mesh {mesh_id}...")
                viewer_base.set_color(color, mesh_id)
                print(f"      set_color completed")

                # Ensure weights are contiguous
                print(f"      Preparing weights for mesh {mesh_id}...")
                Wp_vis = np.ascontiguousarray(fish['Wp'], dtype=np.float64)
                Ws_vis = np.ascontiguousarray(fish['Ws'], dtype=np.float64)
                print(f"      Wp shape: {Wp_vis.shape}, Ws shape: {Ws_vis.shape}")

                print(f"      Calling set_weights for mesh {mesh_id}...")
                viewer_base.set_weights(Wp_vis, Ws_vis, mesh_id)
                print(f"      set_weights completed")

                # Note: init_buffers() is called automatically by launch() via init_all_shaders()
                # We should NOT call it here because it requires an OpenGL context which doesn't exist yet

                print(f"    Mesh {mesh_id} setup complete")
            except Exception as e:
                print(f"    ERROR setting up mesh {mesh_id}: {e}")
                import traceback
                traceback.print_exc()
                raise
        else:
            try:
                print(f"      Reading texture OBJ for mesh {mesh_id}...")
                [Vf, TC, N, Ff, FTC, FN] = fcd.readOBJ_tex(fish['texture_obj'])
                print(f"      Texture OBJ read complete")

                if fish['so'] is not None:
                    Vf = Vf * fish['so']
                if fish['to'] is not None:
                    Vf = Vf - fish['to']
                # Don't apply position offset to texture vertices - position comes from transform
                # Vf stays centered, the transform will position it

                # Prolongation should use centered geometry (weights are computed on centered geometry)
                print(f"      Computing prolongation for mesh {mesh_id}...")
                P = fcd.prolongation(Vf, fish['V_centered'], fish['T'])
                print(f"      Prolongation complete")

                # Ensure arrays are contiguous
                Vf = np.ascontiguousarray(Vf, dtype=np.float64)
                Ff = np.ascontiguousarray(Ff, dtype=np.int32)

                print(f"      Calling set_mesh for textured mesh {mesh_id}...")
                viewer_base.set_mesh(Vf, Ff, mesh_id)
                print(f"      set_mesh completed")

                print(f"      Calling set_texture for mesh {mesh_id}...")
                viewer_base.set_texture(fish['texture_png'], TC, FTC, mesh_id)
                print(f"      set_texture completed")

                Wp_tex = np.ascontiguousarray(P @ fish['Wp'], dtype=np.float64)
                Ws_tex = np.ascontiguousarray(P @ fish['Ws'], dtype=np.float64)

                print(f"      Calling set_weights for textured mesh {mesh_id}...")
                viewer_base.set_weights(Wp_tex, Ws_tex, mesh_id)
                print(f"      set_weights completed")

                print(f"      Setting viewer options for mesh {mesh_id}...")
                viewer_base.set_show_lines(False, mesh_id)
                viewer_base.set_face_based(False, mesh_id)
                print(f"      Viewer options set")

                # Note: init_buffers() is called automatically by launch() via init_all_shaders()
                # We should NOT call it here because it requires an OpenGL context which doesn't exist yet

                print(f"    Mesh {mesh_id} with texture setup complete")
            except Exception as e:
                print(f"    ERROR setting up textured mesh {mesh_id}: {e}")
                import traceback
                traceback.print_exc()
                raise

    print("All meshes added to viewer")

    # Initialize bone transforms for all fishes before launching viewer
    print("Initializing bone transforms for all fishes...")
    for fish_idx, fish in enumerate(fishes):
        mesh_id = mesh_ids[fish_idx]
        # Get initial rig parameters from T0
        p0 = np.ascontiguousarray(fish['T0'][0:3, :].reshape((12, 1)))
        z0 = np.zeros((num_modes * 12, 1), dtype=np.float64)
        # Set initial bone transforms (will be properly initialized when viewer launches)
        try:
            viewer_base.set_bone_transforms(p0, z0, mesh_id)
            print(f"  Initialized bone transforms for fish {fish_idx + 1} (mesh_id {mesh_id})")
        except Exception as e:
            print(f"  Warning: Could not initialize bone transforms for fish {fish_idx + 1}: {e}")

    # Current active fish (controlled by guizmo)
    active_fish_idx = 0
    transform_mode = "translate"

    # Initialize guizmo for first fish
    T0_active = fishes[active_fish_idx]['T0'].copy()

    # Guizmo callback that always updates the currently active fish
    def guizmo_callback(A):
        nonlocal active_fish_idx
        fishes[active_fish_idx]['T0'] = A

    viewer_base.init_guizmo(True, T0_active, guizmo_callback, transform_mode)

    # Key callback to switch between fishes
    def key_callback(key, modifier):
        nonlocal active_fish_idx, transform_mode, T0_active

        # Switch active fish with number keys
        if key >= ord('1') and key <= ord('9'):
            new_idx = key - ord('1')
            if new_idx < len(fishes):
                active_fish_idx = new_idx
                # Get the current T0 from the active fish and ensure it's properly formatted
                T0_active = fishes[active_fish_idx]['T0'].copy().astype(dtype=np.float32, order="F")

                # Re-initialize guizmo with the new active fish's transform
                # This moves the guizmo to the new fish's position
                viewer_base.init_guizmo(True, T0_active, guizmo_callback, transform_mode)
                print(f"Switched to fish {active_fish_idx + 1} at position {T0_active[0:3, 3]}")
                return True

        # Toggle transform mode
        if key == ord('g') or key == ord('G'):
            if transform_mode == "translate":
                transform_mode = "rotate"
            elif transform_mode == "rotate":
                transform_mode = "scale"
            elif transform_mode == "scale":
                transform_mode = "translate"
            viewer_base.change_guizmo_op(transform_mode)
            return True

        return False

    viewer_base.set_key_callback(key_callback)

    # Pre-draw callback to update all simulations
    step = 0
    def pre_draw_callback():
        nonlocal step, active_fish_idx, T0_active, mesh_ids

        # Update all fishes
        for fish_idx, fish in enumerate(fishes):
            try:
                # Get the correct mesh_id for this fish
                if fish_idx >= len(mesh_ids):
                    print(f"Warning: fish_idx {fish_idx} >= len(mesh_ids) {len(mesh_ids)}")
                    continue

                mesh_id = mesh_ids[fish_idx]

                # Validate mesh_id
                if mesh_id < 0:
                    print(f"Warning: Invalid mesh_id {mesh_id} for fish {fish_idx}")
                    continue

                # Get the current T0 transform
                T0_current = fish['T0'].copy()
                T0_initial = fish['T0_initial']

                # For simulation: compute transform relative to initial T0
                # This ensures both fishes see the same relative changes from their starting positions
                # T0_relative represents what has changed from the initial state
                # Since both start with identity in their state, we compute: T0_relative = T0_initial^(-1) * T0_current
                T0_initial_inv = np.linalg.inv(T0_initial.astype(np.float64))
                T0_relative = (T0_initial_inv @ T0_current.astype(np.float64)).astype(np.float32)

                p_relative = np.ascontiguousarray(T0_relative[0:3, :].reshape((12, 1)), dtype=np.float64)

                # Step simulation with relative transform (both fishes behave identically)
                z = fish['sim'].step(p_relative, fish['st'])
                fish['st'].update(z, p_relative)

                # For rendering, use full T0 (with position offset) so fish appears at correct location
                p = np.ascontiguousarray(fish['T0'][0:3, :].reshape((12, 1)), dtype=np.float64)

                # Ensure z is contiguous before passing to C++
                z_contiguous = np.ascontiguousarray(z)

                # Update viewer using the correct mesh_id
                # Only update if this is one of the first few steps (to avoid spam)
                if step < 3:
                    print(f"  Pre-draw step {step}: Updating fish {fish_idx} (mesh_id {mesh_id})")

                viewer_base.set_bone_transforms(p, z_contiguous, mesh_id)
                viewer_base.updateGL(mesh_id)
            except Exception as e:
                # Don't crash the whole app if one fish has an issue
                print(f"Error updating fish {fish_idx} in pre_draw_callback: {e}")
                import traceback
                traceback.print_exc()
                # Continue with other fishes instead of crashing

        # Sync guizmo to active fish in pre-draw callback
        # This ensures the guizmo stays at the active fish's position during animation
        # Note: We don't re-init here to avoid performance issues, just sync if accessible
        try:
            # Try to access guizmo directly if available (for syncing during animation)
            if hasattr(viewer_base, 'guizmo') and viewer_base.guizmo is not None:
                T0_active = fishes[active_fish_idx]['T0'].copy().astype(dtype=np.float32, order="F")
                viewer_base.guizmo.T = T0_active
        except (AttributeError, TypeError):
            # Guizmo might not be directly accessible, that's okay
            # The position will be updated when switching fishes via key callback
            pass

        step += 1

    viewer_base.set_pre_draw_callback(pre_draw_callback)

    print("=" * 60)
    print("Multi-Fish Interactive CD Simulation")
    print("=" * 60)
    print(f"  Number of fishes: {num_fishes}")
    print("  Controls:")
    print("    1-9        Switch active fish")
    print("    g          Toggle Guizmo transform mode (translate/rotate/scale)")
    print("    c          Toggle secondary motion (if supported)")
    print("=" * 60)

    viewer_base.launch(60, True)

if __name__ == "__main__":
    interactive_cd_multi_fish()
