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
                                            enable_caustics=False,
                                            secondary_motion_scale=1.0, secondary_motion_max=3.0,
                                            active_fish_indices=None, camera_follow_fish=None):
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
    secondary_motion_scale : float
        Scaling factor for secondary motion (default=1.0). Lower values reduce secondary motion intensity.
    secondary_motion_max : float
        Maximum magnitude for secondary motion to prevent excessive deformation (default=10.0).
        Higher values allow more deformation, lower values clamp it more aggressively.

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
    elif isinstance(msh_files, list) and len(msh_files) < num_fishes:
        # If fewer msh_files than num_fishes, repeat the last one or use default
        if len(msh_files) == 0:
            default_msh = fc.get_data("./cd_fish.msh")
            msh_files = [default_msh] * num_fishes
        else:
            # Repeat the last msh_file for remaining fishes
            last_msh = msh_files[-1]
            msh_files = msh_files + [last_msh] * (num_fishes - len(msh_files))

    if Vs is None:
        Vs = []
        Ts = []
        for msh_file in msh_files:
            [V, F, T] = fcd.readMSH(msh_file)
            Vs.append(V)
            Ts.append(T)

        # Safety check: ensure we have enough meshes
        if len(Vs) < num_fishes:
            raise ValueError(f"Not enough meshes loaded: expected {num_fishes}, got {len(Vs)}. "
                          f"Please provide {num_fishes} msh_files or set num_fishes={len(Vs)}")
    elif not isinstance(Vs, list):
        Vs = [Vs] * num_fishes
        Ts = [Ts] * num_fishes
    elif len(Vs) < num_fishes:
        raise ValueError(f"Not enough vertex arrays: expected {num_fishes}, got {len(Vs)}. "
                      f"Please provide {num_fishes} Vs or set num_fishes={len(Vs)}")

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
        print(f"\n{'='*60}")
        print(f"Creating fish {fish_idx + 1}/{num_fishes}")
        print(f"{'='*60}")

        V = Vs[fish_idx].copy()
        T = Ts[fish_idx].copy()

        # Scale and center geometry (compute subspace on centered geometry)
        [V_centered, so, to] = fcd.scale_and_center_geometry(V, 1, np.array([[0, 0, 0.]]))

        # Create primary handle on centered geometry
        Wp = np.ones((V_centered.shape[0], 1))
        J = fc.lbs_jacobian(V_centered, Wp)

        # Use separate cache directory for each fish to avoid matrix mismatch errors
        # Each fish may have different geometry, so they need separate eigenmode caches
        # This prevents "Factor is exactly singular" errors when different meshes share cache
        fish_cache_dir = os.path.join(cache_dir, f"fish_{fish_idx}")

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
            # Each fish uses its own cache, so we can read cache if available
            # Fish 0 writes cache, other fishes can read their own cache if it exists
            use_read_cache = read_cache
            sim = fc.fast_cd_sim(V_sim, T_sim, B_sim, l_sim, J_sim,
                                mu=mu, rho=rho, h=1e-2,
                                cache_dir=fish_cache_dir,
                                read_cache=use_read_cache,
                                write_cache=True)  # All fishes can write their own cache
            print(f"  Simulation created successfully for fish {fish_idx + 1}")
        except Exception as e:
            print(f"Error creating simulation for fish {fish_idx}: {e}")
            import traceback
            traceback.print_exc()
            raise

        # Initialize T0 with position offset for visualization
        T0 = np.identity(4).astype(dtype=np.float32, order="F")
        T0[0:3, 3] = fish_positions[fish_idx]

        # Apply initial rotations if specified
        # Fish 0: rotate around Y axis by 1.20 radians (68.69 degrees)
        # Fish 2 and 6 (model 20251121_121348): rotate around X axis by 1.50 radians (86.22 degrees)
        if fish_idx == 0:
            # Rotation around Y axis
            angle_y = 1.20  # 68.69 degrees
            c, s = np.cos(angle_y), np.sin(angle_y)
            R_y = np.array([[c, 0, s],
                           [0, 1, 0],
                           [-s, 0, c]], dtype=np.float32)
            T0[0:3, 0:3] = R_y @ T0[0:3, 0:3]
        elif fish_idx == 2 or fish_idx == 6:  # Fish 2 and 6 use model from 20251121_121348
            # Rotation around X axis
            angle_x = 1.50  # 86.22 degrees
            c, s = np.cos(angle_x), np.sin(angle_x)
            R_x = np.array([[1, 0, 0],
                           [0, c, -s],
                           [0, s, c]], dtype=np.float32)
            T0[0:3, 0:3] = R_x @ T0[0:3, 0:3]

        # Store initial T0 (for reference, but we'll use T0 directly like single fish version)
        T0_initial = T0.copy()

        # Initialize state with initial T0 (including position offset)
        # This matches the single fish behavior where state starts with T0
        # CRITICAL: Each fish must start with its own T0 in the state to prevent exaggerated motion
        z0 = np.zeros((num_modes * 12, 1), dtype=np.float64)
        p0 = T0[0:3, :].reshape((12, 1)).astype(dtype=np.float64)
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
        print(f"  Fish {fish_idx + 1} created with full simulation")

    # Create viewer with multiple meshes
    vertex_shader_path = fc.get_shader("./vertex_shader_16.glsl")
    fragment_shader_path = fc.get_shader("./fragment_shader.glsl")

    viewer_base = fcd.fast_cd_viewer_custom_shader(vertex_shader_path,
                                                   fragment_shader_path, 16, 16)

    # Light position will be set after OpenGL initialization in pre_draw_callback
    # to avoid segfault (OpenGL context must be ready before setting uniforms)
    light_position = np.array([0.0, 5.0, 0.0], dtype=np.float64)
    print(f"  Light position will be set to: {light_position} (after OpenGL init)")

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

    # === UNDERWATER VISUAL EFFECT: GRADIENT BACKGROUND ===
    # Create HUGE background wall positioned far behind the scene
    # This creates the visible gradient effect like in the reference image

    print("\n=== Setting up Underwater Gradient Background ===")

    # Create ONE SINGLE LARGE QUAD for gradient background
    # TEST: Put it close and small to verify rendering
    bg_width = 20.0    # Much smaller for testing
    bg_height = 10.0   # Much smaller for testing
    bg_depth = -5.0    # Close to camera, should be visible

    # Create single quad vertices
    # Gradient will be calculated in shader using screen-space Y coordinate
    bg_V = np.array([
        [-bg_width/2, -bg_height/2, bg_depth],  # Bottom-left
        [bg_width/2, -bg_height/2, bg_depth],   # Bottom-right
        [bg_width/2, bg_height/2, bg_depth],    # Top-right
        [-bg_width/2, bg_height/2, bg_depth],   # Top-left
    ], dtype=np.float64)

    # Create triangles
    bg_F = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)

    # Add background quad as single mesh
    bg_mesh_id = viewer_base.add_mesh()
    viewer_base.set_mesh(bg_V, bg_F, bg_mesh_id)

    # Set up weights (all zero since background doesn't animate)
    num_verts = bg_V.shape[0]
    Wp_bg = np.zeros((num_verts, 16), dtype=np.float64)
    Wp_bg[:, 0] = 1.0
    Ws_bg = np.zeros((num_verts, 16), dtype=np.float64)
    viewer_base.set_weights(Wp_bg, Ws_bg, bg_mesh_id)

    # Set BRIGHT MAGENTA color for debugging - if you see this, the quad is rendering!
    # This helps us verify the mesh is visible before testing the shader gradient
    viewer_base.set_color(np.array([1.0, 0.0, 1.0]), bg_mesh_id)  # Bright magenta/purple

    # Set bone transforms (identity)
    p0_bg = np.zeros((16 * 12, 1), dtype=np.float64)
    for bone_idx in range(16):
        base_idx = bone_idx * 12
        p0_bg[base_idx + 0] = 1.0
        p0_bg[base_idx + 5] = 1.0
        p0_bg[base_idx + 10] = 1.0
    z0_bg = np.zeros((16 * 12, 1), dtype=np.float64)
    viewer_base.set_bone_transforms(p0_bg, z0_bg, bg_mesh_id)

    # Set rendering properties for background quad
    viewer_base.set_face_based(False, bg_mesh_id)  # Try smooth shading
    viewer_base.set_show_lines(True, bg_mesh_id)   # SHOW WIREFRAME for debugging!

    print(f"  Created background gradient: SINGLE QUAD (mesh ID {bg_mesh_id})")
    print(f"  DEBUG: Mesh has {num_verts} vertices")
    print(f"  DEBUG: Positioned at Z={bg_depth}, size {bg_width}x{bg_height}")
    print(f"  DEBUG: Vertices: {bg_V}")
    print(f"  DEBUG: If you see MAGENTA wireframe quad, mesh is rendering!")
    print(f"  Shader will render smooth gradient from dark blue (bottom) to light cyan (top)")

    # Initialize bone transforms for all fishes
    print("\nInitializing bone transforms for all fishes...")
    for fish_idx, fish in enumerate(fishes):
        mesh_id = mesh_ids[fish_idx]
        p0 = np.ascontiguousarray(fish['T0'][0:3, :].reshape((12, 1)))
        z0 = np.zeros((num_modes * 12, 1), dtype=np.float64)
        try:
            viewer_base.set_bone_transforms(p0, z0, mesh_id)
            print(f"  Initialized bone transforms for fish {fish_idx + 1} (mesh_id {mesh_id})")
        except Exception as e:
            print(f"  Warning: Could not initialize bone transforms for fish {fish_idx + 1}: {e}")

    # Set background color to match skybox bottom (deep dark blue)
    # This matches the shader's skyBottom color: vec3(0.02, 0.12, 0.25)
    background_color = np.array([5, 31, 64]) / 255.0
    viewer_base.set_background_color(background_color)

    # === ADD OCEAN FLOOR (3x3 GRID for better performance) ===
    floor_ids_ref = []
    floor_transforms_ref = {}
    floor_tiles = []  # Will store floor tile data for treadmill technique
    floor_treadmill_data = [{  # Shared data for treadmill
        'tile_spacing_x': 0.0,
        'tile_spacing_z': 0.0,
        'floor_y_offset': 0.0
    }]

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
        floor_texture_path = fc.get_data("sandtexture-light.png")
        if not os.path.exists(floor_texture_path):
            raise FileNotFoundError
    except:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
        floor_texture_path = os.path.join(project_root, "data", "sandtexture-light.png")

    if os.path.exists(floor_path1) or os.path.exists(floor_path2):
        try:
            print(f"\nLoading ocean floor meshes for randomized 3x3 grid...")

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
            print(f"  Creating 3x3 grid (9 tiles) with randomized floor meshes and overlap...")

            # Create 3x3 grid of floor tiles (reduced from 5x5 for performance)
            num_bones = 16
            num_modes_floor = 16

            overlap_factor = 0.9
            tile_spacing_x = tile_width * overlap_factor
            tile_spacing_z = tile_depth * overlap_factor

            # Store treadmill data for later use
            floor_treadmill_data[0]['tile_spacing_x'] = tile_spacing_x
            floor_treadmill_data[0]['tile_spacing_z'] = tile_spacing_z
            floor_treadmill_data[0]['floor_y_offset'] = floor_y_offset

            random.seed(42)

            for i in range(3):  # rows (reduced from 5)
                for j in range(3):  # columns (reduced from 5)
                    V_floor_base, F_floor, TC_floor, FTC_floor, mesh_name = random.choice(floor_options)

                    x_offset = (j - 1) * tile_spacing_x  # Changed from (j-2) to (j-1) for 3x3 grid
                    z_offset = (i - 1) * tile_spacing_z  # Changed from (i-2) to (i-1) for 3x3 grid

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
                    # Store: (floor_id, grid_i, grid_j, original_vertices, faces, texture_coords, texture_faces)
                    floor_tiles.append((floor_id, i, j, V_floor_scaled.copy(), F_floor, TC_floor, FTC_floor, mesh_name))

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

    # === ADD DECORATIVE ROCKS AROUND THE FLOOR ===
    rock_ids_ref = []
    rock_transforms_ref = {}
    try:
        # Load rock model from Archive - use fast_cody module path for reliability
        import fast_cody
        fc_module_dir = os.path.dirname(fast_cody.__file__)
        # Go up two levels: fast_cody module -> src -> fast_cody project root
        fc_src_dir = os.path.dirname(fc_module_dir)
        fc_project_root = os.path.dirname(fc_src_dir)
        rock_path = os.path.join(fc_project_root, "data", "Archive", "4400rocks.obj")

        print(f"\n=== DEBUG: Rock Loading ===")
        print(f"  fast_cody module at: {fc_module_dir}")
        print(f"  Looking for rocks at: {rock_path}")
        print(f"  Rock file exists: {os.path.exists(rock_path)}")

        if os.path.exists(rock_path):
            print(f"\n=== Adding Decorative Rocks (same as floor loading) ===")
            print(f"Loading rocks from: {rock_path}")

            # Load rock mesh (same as floor)
            [V_rock_base, TC_rock, N_rock, F_rock, FTC_rock, FN_rock] = fcd.readOBJ_tex(rock_path)
            print(f"  Rock mesh loaded: {V_rock_base.shape[0]} vertices, {F_rock.shape[0]} faces")

            # Scale rocks same as floor tiles
            rock_scale = 0.05  # Same as floor tiles
            floor_y_offset = -0.5  # Floor surface level
            rock_y_offset = -0.4  # Rocks sit clearly ON TOP of floor

            # Define rock position - single decorative rock on the sea floor
            rock_positions = [
                [0.0, rock_y_offset, -3.0],     # Center
            ]

            num_bones = 16
            num_modes_rock = 16

            for idx, pos in enumerate(rock_positions):
                # Scale and position rock (EXACTLY like floor)
                V_rock_scaled = V_rock_base * rock_scale
                V_rock_tile = V_rock_scaled.copy()
                V_rock_tile[:, 0] += pos[0]
                V_rock_tile[:, 1] += pos[1]
                V_rock_tile[:, 2] += pos[2]
                V_rock_tile = np.ascontiguousarray(V_rock_tile, dtype=np.float64)

                # Add rock mesh to viewer (EXACTLY like floor)
                rock_id = viewer_base.add_mesh()
                viewer_base.set_mesh(V_rock_tile, F_rock, rock_id)

                # Set rock color (gray/brown stone color) - single color for the mesh
                rock_color = np.array([[0.47, 0.43, 0.39]])  # Row vector [1, 3] format required
                viewer_base.set_color(rock_color, rock_id)
                viewer_base.set_face_based(True, rock_id)
                viewer_base.invert_normals(True, rock_id)  # SAME as floor!
                viewer_base.set_show_lines(False, rock_id)

                # Get vertex count for weights setup
                num_verts = V_rock_tile.shape[0]

                # Set up weights (EXACTLY like floor)
                Wp_rock = np.zeros((num_verts, num_bones), dtype=np.float64)
                Wp_rock[:, 0] = 1.0
                Ws_rock = np.zeros((num_verts, num_modes_rock), dtype=np.float64)
                viewer_base.set_weights(Wp_rock, Ws_rock, rock_id)

                # Set identity bone transforms (EXACTLY like floor)
                p0_rock = np.zeros((num_bones * 12, 1), dtype=np.float64)
                for bone_idx in range(num_bones):
                    base_idx = bone_idx * 12
                    p0_rock[base_idx + 0] = 1.0
                    p0_rock[base_idx + 5] = 1.0
                    p0_rock[base_idx + 10] = 1.0
                z0_rock = np.zeros((num_modes_rock * 12, 1), dtype=np.float64)

                viewer_base.set_bone_transforms(p0_rock, z0_rock, rock_id)

                rock_ids_ref.append(rock_id)
                rock_transforms_ref[rock_id] = (p0_rock, z0_rock)  # Store like floor!
                print(f"  Rock {idx + 1}: mesh ID {rock_id}, position={pos}, vertices={num_verts}, color={rock_color}")

            print(f"\n✓ Rocks loaded successfully: {len(rock_positions)} rocks")
            print(f"  Rock scale: {rock_scale}, Y offset: {rock_y_offset}")
        else:
            print(f"\n  Rock model not found at {rock_path}, skipping rocks")
    except Exception as e:
        print(f"\n  ERROR loading rocks: {e}")
        import traceback
        traceback.print_exc()
        print("  Continuing without rocks...")

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
        # Update the active fish's T0 when guizmo is manipulated
        # Ensure we're updating the correct fish and preserving the transform correctly
        fishes[active_fish_idx]['T0'] = A.copy().astype(dtype=np.float32, order="F")
        # Debug: print when guizmo updates (only occasionally to avoid spam)
        if hasattr(guizmo_callback, 'last_print_step'):
            if step - guizmo_callback.last_print_step > 60:
                print(f"  [Fish {active_fish_idx + 1}] Guizmo updated T0: position={A[0:3, 3]}")
                guizmo_callback.last_print_step = step
        else:
            guizmo_callback.last_print_step = 0

    viewer_base.init_guizmo(True, T0_active, guizmo_callback, transform_mode)

    # Key callback to switch between fishes
    def key_callback(key, modifier):
        nonlocal active_fish_idx, transform_mode, T0_active, camera_mode

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

        # Toggle camera mode (third-person / first-person)
        if key == ord('v') or key == ord('V'):
            if camera_mode[0] == 'third_person':
                camera_mode[0] = 'first_person'
                print(f"Camera mode: FIRST-PERSON (following fish {active_fish_idx + 1})")
            else:
                camera_mode[0] = 'third_person'
                print(f"Camera mode: THIRD-PERSON (static overview)")
            return True

        # Save fish positions to file
        if key == ord('s') or key == ord('S'):
            save_path = "fish_positions.json"
            try:
                positions_data = []
                for fish_idx, fish in enumerate(fishes):
                    T0 = fish['T0']
                    position = T0[0:3, 3].tolist()  # Extract position from transform
                    rotation = T0[0:3, 0:3].tolist()  # Extract rotation matrix
                    positions_data.append({
                        'fish_idx': fish_idx,
                        'position': position,
                        'rotation': rotation
                    })

                import json
                with open(save_path, 'w') as f:
                    json.dump(positions_data, f, indent=2)
                print(f"\n✓ Saved positions of {len(fishes)} fish to {save_path}")
                return True
            except Exception as e:
                print(f"\n✗ Failed to save positions: {e}")
                return False

        # Load fish positions from file
        if key == ord('l') or key == ord('L'):
            load_path = "fish_positions.json"
            try:
                import json
                import os
                if not os.path.exists(load_path):
                    print(f"\n✗ No saved positions found at {load_path}")
                    return False

                with open(load_path, 'r') as f:
                    positions_data = json.load(f)

                # Update fish positions
                loaded_count = 0
                skipped_count = 0
                for data in positions_data:
                    fish_idx = data['fish_idx']
                    if fish_idx < len(fishes):
                        position = np.array(data['position'], dtype=np.float32)
                        rotation = np.array(data['rotation'], dtype=np.float32)

                        # Reconstruct T0 transform
                        T0_new = fishes[fish_idx]['T0'].copy()
                        T0_new[0:3, 0:3] = rotation
                        T0_new[0:3, 3] = position
                        fishes[fish_idx]['T0'] = T0_new
                        loaded_count += 1
                    else:
                        skipped_count += 1

                # Update active fish guizmo
                T0_active = fishes[active_fish_idx]['T0'].copy().astype(dtype=np.float32, order="F")
                viewer_base.init_guizmo(True, T0_active, guizmo_callback, transform_mode)

                print(f"\n✓ Loaded positions for {loaded_count} fish from {load_path}")
                if skipped_count > 0:
                    print(f"  Note: {skipped_count} saved positions skipped (not enough fish in current scene)")
                if len(fishes) > len(positions_data):
                    print(f"  Note: {len(fishes) - len(positions_data)} fish not in saved file (using default positions)")
                return True
            except Exception as e:
                print(f"\n✗ Failed to load positions: {e}")
                import traceback
                traceback.print_exc()
                return False

        return False

    viewer_base.set_key_callback(key_callback)

    # === BUBBLE PARTICLE SYSTEM ===
    num_bubbles = 30  # Reduced for performance (was 100)
    bubble_speed = 0.3  # units per second
    bubble_spawn_y = -0.5  # ocean floor level
    bubble_max_y = 5.0  # respawn when bubbles reach this height
    bubble_spawn_range_x = (-4.0, 4.0)  # spawn range in X
    bubble_spawn_range_z = (-4.0, 4.0)  # spawn range in Z
    bubble_radius = 0.03  # radius of bubble spheres

    # Create a smooth sphere mesh for bubbles using UV sphere subdivision
    def create_bubble_mesh(radius, resolution=8):
        """Create a smooth sphere mesh using UV sphere (latitude/longitude subdivision)"""
        vertices = []
        faces = []

        # Generate vertices using spherical coordinates
        for i in range(resolution + 1):  # latitude
            theta = i * np.pi / resolution  # 0 to pi
            sin_theta = np.sin(theta)
            cos_theta = np.cos(theta)

            for j in range(resolution):  # longitude
                phi = j * 2 * np.pi / resolution  # 0 to 2*pi
                x = radius * sin_theta * np.cos(phi)
                y = radius * cos_theta
                z = radius * sin_theta * np.sin(phi)
                vertices.append([x, y, z])

        # Generate faces (triangles)
        for i in range(resolution):
            for j in range(resolution):
                # Current quad vertices
                v0 = i * resolution + j
                v1 = i * resolution + (j + 1) % resolution
                v2 = (i + 1) * resolution + (j + 1) % resolution
                v3 = (i + 1) * resolution + j

                # Split quad into two triangles
                faces.append([v0, v1, v2])
                faces.append([v0, v2, v3])

        V = np.array(vertices, dtype=np.float64)
        F = np.array(faces, dtype=np.int32)
        return V, F

    # Create base bubble mesh (smooth sphere with 8x8 resolution)
    V_sphere, F_sphere = create_bubble_mesh(bubble_radius, resolution=8)

    # Initialize bubble positions and velocities
    bubbles = []
    bubble_mesh_ids = []
    for i in range(num_bubbles):
        bubble = {
            'pos': np.array([
                random.uniform(bubble_spawn_range_x[0], bubble_spawn_range_x[1]),
                bubble_spawn_y + random.uniform(0, 0.5),  # slight random height offset
                random.uniform(bubble_spawn_range_z[0], bubble_spawn_range_z[1])
            ]),
            'speed': bubble_speed * random.uniform(0.7, 1.3),  # random speed variation
        }
        bubbles.append(bubble)

        # Create a mesh for this bubble
        bubble_id = viewer_base.add_mesh()
        bubble_mesh_ids.append(bubble_id)

        # Set sphere mesh at bubble position
        V_bubble = V_sphere.copy()
        V_bubble += bubble['pos']
        viewer_base.set_mesh(V_bubble, F_sphere, bubble_id)

        # Set bubble color (light blue-white)
        bubble_color = np.array([200, 240, 255]) / 255.0
        viewer_base.set_color(bubble_color, bubble_id)
        viewer_base.set_face_based(True, bubble_id)
        viewer_base.set_show_lines(False, bubble_id)

        # Set up weights and transforms for bubble (static, no animation)
        num_verts = V_bubble.shape[0]
        Wp_bubble = np.zeros((num_verts, 16), dtype=np.float64)
        Wp_bubble[:, 0] = 1.0
        Ws_bubble = np.zeros((num_verts, 16), dtype=np.float64)
        viewer_base.set_weights(Wp_bubble, Ws_bubble, bubble_id)

        # Set identity transform
        p0_bubble = np.zeros((16 * 12, 1), dtype=np.float64)
        for bone_idx in range(16):
            base_idx = bone_idx * 12
            p0_bubble[base_idx + 0] = 1.0
            p0_bubble[base_idx + 5] = 1.0
            p0_bubble[base_idx + 10] = 1.0
        z0_bubble = np.zeros((16 * 12, 1), dtype=np.float64)
        viewer_base.set_bone_transforms(p0_bubble, z0_bubble, bubble_id)

    # Pre-draw callback to update all simulations
    step = 0
    caustics_initialized = [False]  # Use list to make it mutable in nested function
    light_initialized = [False]  # Use list to make it mutable in nested function
    camera_initialized = [False]  # Use list to make it mutable in nested function
    camera_mode = ['third_person']  # 'third_person' or 'first_person'

    # Treadmill technique: track last grid position to update floor tiles
    last_grid_pos = [np.array([0.0, 0.0])]  # [x_grid, z_grid]

    def pre_draw_callback():
        nonlocal step, active_fish_idx, T0_active, mesh_ids, bubbles, light_position, camera_mode, last_grid_pos

        # Initialize camera position on first frame to see all fishes
        if not camera_initialized[0]:
            try:
                print("\n=== Initializing Camera Position (frame 0) ===")
                if hasattr(viewer_base, 'set_camera_eye') and hasattr(viewer_base, 'set_camera_center'):
                    # Calculate scene bounds to position camera appropriately
                    # Fishes span: X[-5.6, 4.3], Y[0.4, 5.0], Z[-6.2, 0.7]
                    # Set camera high up and far back to see all fishes
                    camera_eye = np.array([[0.0, 10.0, 12.0]])  # High up and far back (row vector)
                    camera_center = np.array([[0.0, 2.5, -3.0]])  # Look at center of fish cluster (row vector)

                    viewer_base.set_camera_eye(camera_eye)
                    viewer_base.set_camera_center(camera_center)

                    print(f"  Camera position set: eye=[0, 10, 12], center=[0, 2.5, -3]")
                    print(f"  All {num_fishes} fishes and floor should be visible")
                    print(f"  Press 'v' to toggle between third-person and first-person camera")
                    camera_initialized[0] = True
                    print("=== Camera Initialization Complete ===\n")
                else:
                    print("  WARNING: set_camera methods not available, using default camera")
                    camera_initialized[0] = True
            except Exception as e:
                print(f"  ERROR initializing camera: {e}")
                import traceback
                traceback.print_exc()
                camera_initialized[0] = True  # Don't try again

        # Initialize light position on first frame (after OpenGL context is ready)
        if not light_initialized[0]:
            try:
                print("\n=== Initializing Light Position (frame 0) ===")
                if hasattr(viewer_base, 'set_light_position'):
                    viewer_base.set_light_position(light_position)
                    print(f"  Light position set to: {light_position}")
                    print(f"  Underwater gradient enabled in shader (world-space Y)")
                    light_initialized[0] = True
                    print("=== Light Initialization Complete ===\n")
                else:
                    print("  ERROR: set_light_position method not available")
                    light_initialized[0] = True  # Don't try again
            except Exception as e:
                print(f"  ERROR initializing light: {e}")
                import traceback
                traceback.print_exc()
                light_initialized[0] = True  # Don't try again

        # Initialize caustics on first frame (after OpenGL context is ready)
        if enable_caustics and not caustics_initialized[0] and caustics_atlas_path:
            try:
                print("\n=== Initializing Caustics (frame 0) ===")
                if hasattr(viewer_base, 'set_caustics_atlas'):
                    print(f"  Loading caustics atlas from: {caustics_atlas_path}")
                    viewer_base.set_caustics_atlas(caustics_atlas_path)
                    print(f"  Caustics atlas loaded successfully")

                    # Set shader uniforms for all meshes (fish + floor + rocks)
                    if hasattr(viewer_base, 'set_uniform'):
                        all_mesh_ids = mesh_ids + floor_ids_ref + rock_ids_ref
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

        # Camera update: ONLY update in first-person mode (let user control third-person manually)
        if camera_mode[0] == 'first_person':
            if hasattr(viewer_base, 'set_camera_eye') and hasattr(viewer_base, 'set_camera_center'):
                try:
                    # Use active_fish_idx instead of camera_follow_fish for following
                    if active_fish_idx < len(fishes):
                        fish_to_follow = fishes[active_fish_idx]
                        fish_position = fish_to_follow['T0'][0:3, 3]  # Extract position from transform

                        # Get fish rotation matrix to orient camera
                        R = fish_to_follow['T0'][0:3, 0:3]
                        forward = R[:, 0]  # Fish's forward direction
                        up = R[:, 2]       # Fish's up direction

                        # Camera positioned behind and above the fish
                        camera_distance = 3.0
                        camera_height = 1.0
                        camera_eye = fish_position - forward * camera_distance + up * camera_height

                        # Look ahead of the fish
                        look_ahead = 1.5
                        camera_center = fish_position + forward * look_ahead

                        viewer_base.set_camera_center(np.array([camera_center]).reshape(1, 3))
                        viewer_base.set_camera_eye(np.array([camera_eye]).reshape(1, 3))
                except Exception as e:
                    if step % 60 == 0:  # Print error occasionally
                        print(f"Camera update failed: {e}")

        # Update all fishes
        for fish_idx, fish in enumerate(fishes):
            try:
                if fish_idx >= len(mesh_ids):
                    continue

                mesh_id = mesh_ids[fish_idx]

                if mesh_id < 0:
                    continue

                # Get current T0 transform (updated by guizmo callback when manipulated)
                T0_current = fish['T0'].copy()

                # CRITICAL FIX: Use T0 directly for simulation (like single fish version)
                # The position offset in T0 should not affect simulation if state is initialized correctly
                # Each fish's simulation state starts at identity, so using T0 directly should work
                # The key is that the simulation sees the absolute transform, not relative
                p = np.ascontiguousarray(T0_current[0:3, :].reshape((12, 1)), dtype=np.float64)

                # Check if this fish should have active simulation (secondary motion)
                # If active_fish_indices is None, all fish are active
                # If active_fish_indices is provided, only those fish have secondary motion
                has_secondary_motion = (active_fish_indices is None) or (fish_idx in active_fish_indices)

                # Step simulation with relative transform (prevents position offset from affecting simulation)
                if has_secondary_motion:
                    z = fish['sim'].step(p, fish['st'])
                else:
                    # No secondary motion - use zero vector
                    z = np.zeros_like(fish['st'].z)

                # Apply scaling to secondary motion
                z_scaled = z * secondary_motion_scale

                # DEBUG: Print secondary motion magnitude every 60 frames
                if step % 60 == 0:
                    z_original_norm = np.linalg.norm(z)
                    motion_status = "ACTIVE" if has_secondary_motion else "STATIC (no secondary motion)"
                    print(f"  [Fish {fish_idx + 1}] {motion_status}: z_norm={z_original_norm:.4f}, scale={secondary_motion_scale}")

                # NO CLAMPING - let all fish have natural secondary motion
                # Update state with secondary motion
                fish['st'].update(z_scaled, p)

                # For rendering, use full T0 (with position offset) so fish appears at correct location
                # Use the scaled secondary motion (no clamping)
                p_render = np.ascontiguousarray(T0_current[0:3, :].reshape((12, 1)), dtype=np.float64)
                z_contiguous = np.ascontiguousarray(z_scaled)

                viewer_base.set_bone_transforms(p_render, z_contiguous, mesh_id)
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

            # Set time uniform for all meshes (fish + floor + rocks)
            all_mesh_ids = mesh_ids + floor_ids_ref + rock_ids_ref
            for mesh_id in all_mesh_ids:
                viewer_base.set_uniform("u_time", float(current_time), mesh_id)

        # Update all floor tiles
        for floor_id in floor_ids_ref:
            if floor_id in floor_transforms_ref:
                p0_floor, z0_floor = floor_transforms_ref[floor_id]
                viewer_base.set_bone_transforms(p0_floor, z0_floor, floor_id)
                viewer_base.updateGL(floor_id)

        # Update all rocks (EXACTLY like floor tiles)
        for rock_id in rock_ids_ref:
            if rock_id in rock_transforms_ref:
                p0_rock, z0_rock = rock_transforms_ref[rock_id]
                viewer_base.set_bone_transforms(p0_rock, z0_rock, rock_id)
                viewer_base.updateGL(rock_id)

        # Update bubble particles
        import time as time_module
        if not hasattr(pre_draw_callback, 'last_bubble_update_time'):
            pre_draw_callback.last_bubble_update_time = time_module.time()

        current_time = time_module.time()
        dt = current_time - pre_draw_callback.last_bubble_update_time
        pre_draw_callback.last_bubble_update_time = current_time

        # Update bubble positions (float upward) and update meshes
        for i, bubble in enumerate(bubbles):
            # Move bubble upward
            bubble['pos'][1] += bubble['speed'] * dt

            # Respawn if bubble reaches max height
            if bubble['pos'][1] > bubble_max_y:
                bubble['pos'] = np.array([
                    random.uniform(bubble_spawn_range_x[0], bubble_spawn_range_x[1]),
                    bubble_spawn_y + random.uniform(0, 0.2),
                    random.uniform(bubble_spawn_range_z[0], bubble_spawn_range_z[1])
                ])
                bubble['speed'] = bubble_speed * random.uniform(0.7, 1.3)

            # Update bubble mesh position
            bubble_id = bubble_mesh_ids[i]
            V_bubble = V_sphere.copy()
            V_bubble += bubble['pos']
            viewer_base.set_vertices(V_bubble, bubble_id)
            viewer_base.updateGL(bubble_id)

        # === INFINITE FLOOR: Treadmill Technique ===
        # Reposition floor tiles to follow camera and create infinite appearance
        if hasattr(viewer_base, 'igl_v') and len(floor_tiles) > 0:
            try:
                # Get treadmill data
                tile_spacing_x = floor_treadmill_data[0]['tile_spacing_x']
                tile_spacing_z = floor_treadmill_data[0]['tile_spacing_z']
                floor_y_offset = floor_treadmill_data[0]['floor_y_offset']

                if tile_spacing_x > 0 and tile_spacing_z > 0:
                    # Get camera position (X, Z)
                    camera_eye = viewer_base.igl_v.core().camera_eye
                    camera_x, camera_z = float(camera_eye[0]), float(camera_eye[2])

                    # Calculate which "grid tile" the camera is in
                    # Snap to multiples of tile_spacing
                    grid_x = np.floor(camera_x / tile_spacing_x)
                    grid_z = np.floor(camera_z / tile_spacing_z)

                    # Check if camera moved to a new grid tile
                    if grid_x != last_grid_pos[0][0] or grid_z != last_grid_pos[0][1]:
                        # Update floor tile positions
                        for floor_data in floor_tiles:
                            floor_id, i, j, V_orig, F, TC, FTC, mesh_name = floor_data

                            # Calculate new offset: keep 3x3 grid centered on camera grid position
                            new_x_offset = (j - 1) * tile_spacing_x + grid_x * tile_spacing_x
                            new_z_offset = (i - 1) * tile_spacing_z + grid_z * tile_spacing_z

                            # Apply new offset to original vertices
                            V_updated = V_orig.copy()
                            V_updated[:, 0] += new_x_offset
                            V_updated[:, 1] += floor_y_offset
                            V_updated[:, 2] += new_z_offset
                            V_updated = np.ascontiguousarray(V_updated, dtype=np.float64)

                            # Update mesh vertices
                            viewer_base.set_vertices(V_updated, floor_id)
                            viewer_base.updateGL(floor_id)

                        # Update last grid position
                        last_grid_pos[0] = np.array([grid_x, grid_z])
                        print(f"[Treadmill] Floor repositioned to grid ({grid_x:.0f}, {grid_z:.0f})")
            except Exception as e:
                if step == 0:  # Only print on first frame
                    print(f"Treadmill update skipped: {e}")

        # Sync guizmo to active fish ONLY if it's not being actively manipulated
        # This prevents overwriting user input during dragging
        # The guizmo callback handles updates during manipulation, so we only sync
        # when the guizmo is not being actively used (to keep it in sync with simulation)
        try:
            if hasattr(viewer_base, 'guizmo') and viewer_base.guizmo is not None:
                # Only sync if guizmo is not being actively manipulated
                # Check if guizmo is being used by checking if it's visible and not in a drag state
                # For now, we'll sync but this should be done carefully to avoid conflicts
                # The guizmo callback should be the source of truth during manipulation
                T0_from_fish = fishes[active_fish_idx]['T0'].copy().astype(dtype=np.float32, order="F")
                T0_from_guizmo = viewer_base.guizmo.T.copy()

                # Only sync if there's a significant difference (guizmo might be slightly out of sync)
                # This prevents overwriting during active manipulation
                diff = np.linalg.norm(T0_from_fish - T0_from_guizmo)
                if diff > 0.01:  # Only sync if difference is significant (0.01 units)
                    # Sync guizmo to fish T0 (fish T0 is source of truth from simulation)
                    viewer_base.guizmo.T = T0_from_fish
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
    print(f"  Number of rocks: {len(rock_ids_ref)}")
    print(f"  Total meshes: {len(mesh_ids) + len(floor_ids_ref) + len(rock_ids_ref)}")
    print(f"  Caustics enabled: {enable_caustics}")
    print("  Controls:")
    print("    1-9        Switch active fish")
    print("    g          Toggle Guizmo transform mode (translate/rotate/scale)")
    print("    v          Toggle camera mode (third-person / first-person)")
    print("    s          Save current fish positions to fish_positions.json")
    print("    l          Load fish positions from fish_positions.json")
    print("    Mouse      Drag to rotate, scroll to zoom")
    print("=" * 60)

    # Set initial camera position BEFORE launching viewer
    print("\nSetting initial camera position...")
    try:
        # Try multiple approaches to set the camera
        if hasattr(viewer_base, 'igl_v'):
            # Direct access to igl viewer core
            core = viewer_base.igl_v.core()
            # Position camera high above to see all fishes
            # Fishes span: X[-5.6, 4.3], Y[0.4, 5.0], Z[-6.2, 0.7]
            core.camera_eye = np.array([0.0, 15.0, 15.0], dtype=np.float32)
            core.camera_center = np.array([0.0, 2.0, -2.5], dtype=np.float32)
            core.camera_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)

            # Don't set zoom values - let libigl handle zoom naturally with scroll wheel
            # The single-fish version doesn't set zoom and works fine

            print(f"  Camera set via igl_v.core(): eye=[0, 15, 15], center=[0, 2, -2.5]")
        elif hasattr(viewer_base, 'set_camera_eye') and hasattr(viewer_base, 'set_camera_center'):
            # Use the custom methods
            initial_eye = np.array([[0.0, 15.0, 15.0]])
            initial_center = np.array([[0.0, 2.0, -2.5]])
            viewer_base.set_camera_eye(initial_eye)
            viewer_base.set_camera_center(initial_center)
            print(f"  Camera set via methods: eye=[0, 15, 15], center=[0, 2, -2.5]")
        else:
            print("  Warning: No camera methods available")
    except Exception as e:
        print(f"  Warning: Could not set initial camera: {e}")
        import traceback
        traceback.print_exc()

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
