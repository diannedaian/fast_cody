import os
import numpy as np
import scipy as sp
import igl
import json
import random
from os.path import basename, splitext

import fast_cd_pyb as fcd
import fast_cody as fc


def load_floor_mesh(floor_path):
    """
    Load and process a floor mesh (OBJ file).
    Returns: (V, F, TC, FTC) or None if failed
    """
    if not os.path.exists(floor_path):
        return None

    try:
        [V_floor, TC_floor, N_floor, F_floor, FTC_floor, FN_floor] = fcd.readOBJ_tex(floor_path)

        # Triangulate quads if needed
        if F_floor.shape[1] == 4:
            num_quads = F_floor.shape[0]
            F_floor_tri = np.zeros((num_quads * 2, 3), dtype=np.int32)
            F_floor_tri[0::2, :] = F_floor[:, [0, 1, 2]]
            F_floor_tri[1::2, :] = F_floor[:, [0, 2, 3]]
            F_floor = F_floor_tri

            if FTC_floor is not None and FTC_floor.shape[1] == 4:
                FTC_tri = np.zeros((num_quads * 2, 3), dtype=np.int32)
                FTC_tri[0::2, :] = FTC_floor[:, [0, 1, 2]]
                FTC_tri[1::2, :] = FTC_floor[:, [0, 2, 3]]
                FTC_floor = FTC_tri

        F_floor = np.ascontiguousarray(F_floor, dtype=np.int32)
        return (V_floor, F_floor, TC_floor, FTC_floor)
    except Exception as e:
        print(f"  Warning: Failed to load {floor_path}: {e}")
        return None


def interactive_cd_affine_handle_multi_fish(msh_files=None, Vs=None, Ts=None, Ws_list=None, l_list=None,
                                            mu=1e4, rho=1e3, num_modes=16, num_clusters=100,
                                            constraint_enforcement="optimal",
                                            cache_dir=None, results_dir=None, read_cache=False,
                                            texture_png_list=None, texture_obj_list=None,
                                            num_fishes=2, fish_positions=None,
                                            enable_caustics=False):
    """
    Runs an interactive fast CD simulation with multiple fishes, floor grid, and optional caustics.
    Each fish can be controlled independently using an affine handle with a Guizmo.

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
    enable_caustics : bool
        If True, adds animated caustics light patterns on the ocean floor (default=False)

    Examples
    --------
    >>> import fast_cody as fc
    >>> fc.apps.interactive_cd_affine_handle_multi_fish()
    >>> fc.apps.interactive_cd_affine_handle_multi_fish(num_fishes=3, enable_caustics=True)
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

    # Set up fish positions - positioned at y=0.0 to be slightly above the floor
    if fish_positions is None:
        fish_positions = []
        for i in range(num_fishes):
            # Position fishes side by side
            offset = (i - (num_fishes - 1) / 2) * 1.5
            fish_positions.append(np.array([offset, 0.0, 0.0]))

    # Process each fish - use separate cache directories to avoid conflicts
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
            C = fc.complementary_constraint_matrix(V_centered, T, J, dt=1e-3)
            C2 = fc.lbs_weight_space_constraint(V_centered, C)
            [B, _, _] = fc.skinning_subspace(V_centered, T, num_modes, num_clusters, C=C2,
                                            read_cache=read_cache,
                                            cache_dir=fish_cache_dir,
                                            constraint_enforcement=constraint_enforcement)

        # Ensure all arrays are contiguous and properly formatted
        V_centered = np.ascontiguousarray(V_centered.copy(), dtype=np.float64)
        T = np.ascontiguousarray(T.copy(), dtype=np.int32)
        B = np.ascontiguousarray(B.copy(), dtype=np.float64)
        l = np.ascontiguousarray(l.copy(), dtype=np.int32).reshape(-1, 1)
        if isinstance(J, sp.sparse.csc_matrix):
            J = J.copy()
        else:
            J = sp.sparse.csc_matrix(J)

        # Create simulation
        import gc
        gc.collect()

        print(f"  Creating simulation for fish {fish_idx + 1}...")

        V_sim = np.ascontiguousarray(V_centered, dtype=np.float64)
        T_sim = np.ascontiguousarray(T, dtype=np.int32)
        B_sim = np.ascontiguousarray(B, dtype=np.float64)
        l_sim = np.ascontiguousarray(l.flatten(), dtype=np.int32)
        J_sim = J.copy() if isinstance(J, sp.sparse.csc_matrix) else sp.sparse.csc_matrix(J)

        try:
            use_read_cache = read_cache or (fish_idx > 0)
            sim = fc.fast_cd_sim(V_sim, T_sim, B_sim, l_sim, J_sim,
                                mu=mu, rho=rho, h=1e-2,
                                cache_dir=fish_cache_dir,
                                read_cache=use_read_cache,
                                write_cache=(fish_idx == 0))
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

        # Initialize state with identity transform
        z0 = np.zeros((num_modes * 12, 1), dtype=np.float64)
        T0_identity = np.identity(4).astype(dtype=np.float32, order="F")
        p0 = T0_identity[0:3, :].reshape((12, 1)).astype(dtype=np.float64)
        st = fc.fast_cd_state(z0, p0)

        # For visualization: use centered vertices
        V_vis = V_centered.copy()

        fishes.append({
            'V': V_vis,
            'V_centered': V_centered,
            'T': T,
            'F': igl.boundary_facets(T)[0],
            'Wp': Wp,
            'Ws': Ws,
            'B': B,
            'l': l,
            'J': J,
            'sim': sim,
            'st': st,
            'T0': T0,
            'T0_initial': T0_initial,
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

    # Add all fish meshes to viewer
    print("Adding meshes to viewer...")
    mesh_ids = []
    for fish_idx in range(num_fishes):
        mesh_id = viewer_base.add_mesh()
        mesh_ids.append(mesh_id)
        print(f"  Created mesh slot {mesh_id} for fish {fish_idx + 1}")

    # Set mesh data for each fish
    for fish_idx, fish in enumerate(fishes):
        mesh_id = mesh_ids[fish_idx]
        print(f"  Setting up mesh {mesh_id} for fish {fish_idx + 1}...")

        vis_texture = (fish['texture_png'] is not None and
                      fish['texture_obj'] is not None)

        if not vis_texture:
            try:
                V_vis = np.ascontiguousarray(fish['V'], dtype=np.float64)
                F_vis = np.ascontiguousarray(fish['F'], dtype=np.int32)

                viewer_base.set_mesh(V_vis, F_vis, mesh_id)
                viewer_base.invert_normals(True, mesh_id)

                # Different colors for different fishes
                colors = [
                    np.array([144, 210, 236]) / 255.0,  # Blue
                    np.array([236, 144, 144]) / 255.0,  # Red
                    np.array([144, 236, 144]) / 255.0,  # Green
                    np.array([236, 236, 144]) / 255.0,  # Yellow
                ]
                color = colors[fish_idx % len(colors)]
                viewer_base.set_color(color, mesh_id)

                Wp_vis = np.ascontiguousarray(fish['Wp'], dtype=np.float64)
                Ws_vis = np.ascontiguousarray(fish['Ws'], dtype=np.float64)
                viewer_base.set_weights(Wp_vis, Ws_vis, mesh_id)

                print(f"    Mesh {mesh_id} setup complete")
            except Exception as e:
                print(f"    ERROR setting up mesh {mesh_id}: {e}")
                import traceback
                traceback.print_exc()
                raise
        else:
            try:
                [Vf, TC, N, Ff, FTC, FN] = fcd.readOBJ_tex(fish['texture_obj'])

                if fish['so'] is not None:
                    Vf = Vf * fish['so']
                if fish['to'] is not None:
                    Vf = Vf - fish['to']

                P = fcd.prolongation(Vf, fish['V_centered'], fish['T'])

                Vf = np.ascontiguousarray(Vf, dtype=np.float64)
                Ff = np.ascontiguousarray(Ff, dtype=np.int32)

                viewer_base.set_mesh(Vf, Ff, mesh_id)
                viewer_base.set_texture(fish['texture_png'], TC, FTC, mesh_id)

                Wp_tex = np.ascontiguousarray(P @ fish['Wp'], dtype=np.float64)
                Ws_tex = np.ascontiguousarray(P @ fish['Ws'], dtype=np.float64)

                viewer_base.set_weights(Wp_tex, Ws_tex, mesh_id)

                viewer_base.set_show_lines(False, mesh_id)
                viewer_base.set_face_based(False, mesh_id)

                print(f"    Mesh {mesh_id} with texture setup complete")
            except Exception as e:
                print(f"    ERROR setting up textured mesh {mesh_id}: {e}")
                import traceback
                traceback.print_exc()
                raise

    print("All fish meshes added to viewer")

    # Initialize bone transforms for all fishes
    print("Initializing bone transforms for all fishes...")
    for fish_idx, fish in enumerate(fishes):
        mesh_id = mesh_ids[fish_idx]
        p0 = np.ascontiguousarray(fish['T0'][0:3, :].reshape((12, 1)))
        z0 = np.zeros((num_modes * 12, 1), dtype=np.float64)
        try:
            viewer_base.set_bone_transforms(p0, z0, mesh_id)
            print(f"  Initialized bone transforms for fish {fish_idx + 1} (mesh_id {mesh_id})")
        except Exception as e:
            print(f"  Warning: Could not initialize bone transforms for fish {fish_idx + 1}: {e}")

    # Set background color to ocean blue (#5FBFD2)
    background_color = np.array([95, 191, 210]) / 255.0
    viewer_base.set_background_color(background_color)

    # === ADD OCEAN FLOOR (5x5 GRID) ===
    floor_ids_ref = []
    floor_transforms_ref = {}

    # Load floor meshes
    try:
        floor_path1 = fc.get_data("sea_floor.obj")
        floor_path2 = fc.get_data("sea_floor2.obj")
        if not os.path.exists(floor_path1) or not os.path.exists(floor_path2):
            raise FileNotFoundError
    except:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
        floor_path1 = os.path.join(project_root, "data", "sea_floor.obj")
        floor_path2 = os.path.join(project_root, "data", "sea_floor2.obj")

    try:
        floor_texture_path = fc.get_data("sandtexture.jpg")
        if not os.path.exists(floor_texture_path):
            raise FileNotFoundError
    except:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
        floor_texture_path = os.path.join(project_root, "data", "sandtexture.jpg")

    if os.path.exists(floor_path1) or os.path.exists(floor_path2):
        try:
            print(f"\nLoading ocean floor meshes for randomized 5x5 grid...")

            floor_mesh1 = load_floor_mesh(floor_path1)
            floor_mesh2 = load_floor_mesh(floor_path2)

            floor_options = []
            if floor_mesh1 is not None:
                V1, F1, TC1, FTC1 = floor_mesh1
                print(f"  Loaded sea_floor.obj: {V1.shape[0]} vertices, {F1.shape[0]} faces")
                floor_options.append((V1, F1, TC1, FTC1, "sea_floor.obj"))
            if floor_mesh2 is not None:
                V2, F2, TC2, FTC2 = floor_mesh2
                print(f"  Loaded sea_floor2.obj: {V2.shape[0]} vertices, {F2.shape[0]} faces")
                floor_options.append((V2, F2, TC2, FTC2, "sea_floor2.obj"))

            if len(floor_options) == 0:
                raise Exception("Failed to load any floor meshes")

            # Floor tile parameters
            tile_scale = 0.05
            floor_y_offset = -0.5

            # Calculate tile size
            tile_widths = []
            tile_depths = []
            for V_opt, F_opt, TC_opt, FTC_opt, name in floor_options:
                V_scaled = V_opt * tile_scale
                tile_widths.append(V_scaled[:, 0].max() - V_scaled[:, 0].min())
                tile_depths.append(V_scaled[:, 2].max() - V_scaled[:, 2].min())

            tile_width = max(tile_widths)
            tile_depth = max(tile_depths)

            print(f"  Tile dimensions: width={tile_width:.2f}, depth={tile_depth:.2f}")
            print(f"  Creating 5x5 grid (25 tiles) with randomized floor meshes and overlap...")

            # Create 5x5 grid of floor tiles
            num_bones = 16
            num_modes_floor = 16
            floor_tiles = []

            overlap_factor = 0.9
            tile_spacing_x = tile_width * overlap_factor
            tile_spacing_z = tile_depth * overlap_factor

            random.seed(42)

            for i in range(5):  # rows
                for j in range(5):  # columns
                    V_floor_base, F_floor, TC_floor, FTC_floor, mesh_name = random.choice(floor_options)

                    x_offset = (j - 2) * tile_spacing_x
                    z_offset = (i - 2) * tile_spacing_z

                    V_floor_scaled = V_floor_base * tile_scale
                    V_floor_tile = V_floor_scaled.copy()
                    V_floor_tile[:, 0] += x_offset
                    V_floor_tile[:, 1] += floor_y_offset
                    V_floor_tile[:, 2] += z_offset
                    V_floor_tile = np.ascontiguousarray(V_floor_tile, dtype=np.float64)

                    floor_id = viewer_base.add_mesh()
                    viewer_base.set_mesh(V_floor_tile, F_floor, floor_id)

                    use_texture = (os.path.exists(floor_texture_path) and
                                  TC_floor is not None and FTC_floor is not None and
                                  TC_floor.shape[0] > 0 and FTC_floor.shape[0] > 0)
                    if use_texture:
                        try:
                            TC_contig = np.ascontiguousarray(TC_floor, dtype=np.float64)
                            FTC_contig = np.ascontiguousarray(FTC_floor, dtype=np.int32)
                            viewer_base.set_texture(floor_texture_path, TC_contig, FTC_contig, floor_id)
                        except Exception as e:
                            print(f"  Warning: Could not apply texture to tile ({i},{j}): {e}")

                    num_verts = V_floor_tile.shape[0]
                    Wp_floor = np.zeros((num_verts, num_bones), dtype=np.float64)
                    Wp_floor[:, 0] = 1.0
                    Ws_floor = np.zeros((num_verts, num_modes_floor), dtype=np.float64)
                    viewer_base.set_weights(Wp_floor, Ws_floor, floor_id)

                    if use_texture:
                        viewer_base.set_show_lines(False, floor_id)
                        viewer_base.set_face_based(False, floor_id)
                    else:
                        floor_color = np.array([200, 180, 150]) / 255.0
                        viewer_base.set_color(floor_color, floor_id)
                        viewer_base.set_face_based(True, floor_id)
                        viewer_base.invert_normals(True, floor_id)
                        viewer_base.set_show_lines(False, floor_id)

                    p0_floor = np.zeros((num_bones * 12, 1), dtype=np.float64)
                    for bone_idx in range(num_bones):
                        base_idx = bone_idx * 12
                        p0_floor[base_idx + 0] = 1.0
                        p0_floor[base_idx + 5] = 1.0
                        p0_floor[base_idx + 10] = 1.0
                    z0_floor = np.zeros((num_modes_floor * 12, 1), dtype=np.float64)

                    viewer_base.set_bone_transforms(p0_floor, z0_floor, floor_id)

                    floor_ids_ref.append(floor_id)
                    floor_transforms_ref[floor_id] = (p0_floor, z0_floor)
                    floor_tiles.append((floor_id, i, j, x_offset, z_offset, mesh_name))

                    print(f"  Tile ({i},{j}): mesh ID {floor_id}, offset=({x_offset:.2f}, {z_offset:.2f}), mesh={mesh_name}")

            print(f"\n  Floor grid loaded successfully: {len(floor_tiles)} tiles")

        except Exception as e:
            print(f"ERROR: Failed to load ocean floor: {e}")
            import traceback
            traceback.print_exc()
            print("  Continuing without floor...")
    else:
        print(f"Floor files not found")
        print("Continuing without floor...")

    # === PREPARE CAUSTICS (but don't load yet - needs OpenGL context) ===
    caustics_atlas_path = None
    if enable_caustics:
        caustics_atlas_path = fc.get_data("caustics/caustics_atlas.png")
        if os.path.exists(caustics_atlas_path):
            print(f"\n  Caustics will be loaded after OpenGL context initialization")
            print(f"  Caustics atlas path: {caustics_atlas_path}")
        else:
            print(f"  Warning: Caustics atlas not found at {caustics_atlas_path}")
            enable_caustics = False
            caustics_atlas_path = None

    # Current active fish
    active_fish_idx = 0
    transform_mode = "translate"

    # Initialize guizmo for first fish
    T0_active = fishes[active_fish_idx]['T0'].copy()

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
                T0_active = fishes[active_fish_idx]['T0'].copy().astype(dtype=np.float32, order="F")
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
    caustics_initialized = [False]  # Use list to make it mutable in nested function

    def pre_draw_callback():
        nonlocal step, active_fish_idx, T0_active, mesh_ids

        # Initialize caustics on first frame (after OpenGL context is ready)
        if enable_caustics and not caustics_initialized[0] and caustics_atlas_path:
            try:
                print("\n=== Initializing Caustics (frame 0) ===")
                if hasattr(viewer_base, 'set_caustics_atlas'):
                    print(f"  Loading caustics atlas from: {caustics_atlas_path}")
                    viewer_base.set_caustics_atlas(caustics_atlas_path)
                    print(f"  Caustics atlas loaded successfully")

                    # Set shader uniforms for all meshes (fish + floor)
                    if hasattr(viewer_base, 'set_uniform'):
                        all_mesh_ids = mesh_ids + floor_ids_ref
                        print(f"  Setting uniforms for {len(all_mesh_ids)} meshes...")

                        for i, mesh_id in enumerate(all_mesh_ids):
                            viewer_base.set_uniform("u_numFrames", 16, mesh_id)
                            viewer_base.set_uniform("u_frameRate", 12.0, mesh_id)
                            if i < 3 or i >= len(all_mesh_ids) - 3:  # Only print first/last few
                                print(f"    Mesh {mesh_id}: uniforms set")
                        print(f"  Caustics shader uniforms configured for all meshes")

                    caustics_initialized[0] = True
                    print("=== Caustics Initialization Complete ===\n")
                else:
                    print("  ERROR: set_caustics_atlas method not available")
                    caustics_initialized[0] = True  # Don't try again
            except Exception as e:
                print(f"  ERROR initializing caustics: {e}")
                import traceback
                traceback.print_exc()
                caustics_initialized[0] = True  # Don't try again

        # Update all fishes
        for fish_idx, fish in enumerate(fishes):
            try:
                if fish_idx >= len(mesh_ids):
                    continue

                mesh_id = mesh_ids[fish_idx]

                if mesh_id < 0:
                    continue

                T0_current = fish['T0'].copy()
                T0_initial = fish['T0_initial']

                # Compute relative transform for simulation
                T0_initial_inv = np.linalg.inv(T0_initial.astype(np.float64))
                T0_relative = (T0_initial_inv @ T0_current.astype(np.float64)).astype(np.float32)

                p_relative = np.ascontiguousarray(T0_relative[0:3, :].reshape((12, 1)), dtype=np.float64)

                # Step simulation
                z = fish['sim'].step(p_relative, fish['st'])
                fish['st'].update(z, p_relative)

                # For rendering, use full T0
                p = np.ascontiguousarray(fish['T0'][0:3, :].reshape((12, 1)), dtype=np.float64)
                z_contiguous = np.ascontiguousarray(z)

                viewer_base.set_bone_transforms(p, z_contiguous, mesh_id)
                viewer_base.updateGL(mesh_id)
            except Exception as e:
                print(f"Error updating fish {fish_idx}: {e}")
                import traceback
                traceback.print_exc()

        # Update caustics time uniform if enabled
        if enable_caustics and hasattr(viewer_base, 'set_uniform'):
            import time as time_module
            if not hasattr(pre_draw_callback, 'start_time'):
                pre_draw_callback.start_time = time_module.time()
            current_time = time_module.time() - pre_draw_callback.start_time

            # Set time uniform for all meshes (fish + floor)
            all_mesh_ids = mesh_ids + floor_ids_ref
            for mesh_id in all_mesh_ids:
                viewer_base.set_uniform("u_time", float(current_time), mesh_id)

        # Update all floor tiles
        for floor_id in floor_ids_ref:
            if floor_id in floor_transforms_ref:
                p0_floor, z0_floor = floor_transforms_ref[floor_id]
                viewer_base.set_bone_transforms(p0_floor, z0_floor, floor_id)
                viewer_base.updateGL(floor_id)

        # Sync guizmo to active fish
        try:
            if hasattr(viewer_base, 'guizmo') and viewer_base.guizmo is not None:
                T0_active = fishes[active_fish_idx]['T0'].copy().astype(dtype=np.float32, order="F")
                viewer_base.guizmo.T = T0_active
        except (AttributeError, TypeError):
            pass

        step += 1

    print("\n" + "=" * 60)
    print("Setting pre-draw callback...")
    viewer_base.set_pre_draw_callback(pre_draw_callback)
    print("Pre-draw callback set successfully")

    print("=" * 60)
    print("Multi-Fish Interactive CD Simulation with Floor and Caustics")
    print("=" * 60)
    print(f"  Number of fishes: {num_fishes}")
    print(f"  Number of floor tiles: {len(floor_ids_ref)}")
    print(f"  Total meshes: {len(mesh_ids) + len(floor_ids_ref)}")
    print(f"  Caustics enabled: {enable_caustics}")
    print("  Controls:")
    print("    1-9        Switch active fish")
    print("    g          Toggle Guizmo transform mode (translate/rotate/scale)")
    print("    c          Toggle secondary motion (if supported)")
    print("=" * 60)

    print("\nLaunching viewer (this will initialize OpenGL context)...")
    try:
        viewer_base.launch(60, True)
        print("Viewer closed normally")
    except Exception as e:
        print(f"\nERROR during viewer launch: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    interactive_cd_affine_handle_multi_fish()
