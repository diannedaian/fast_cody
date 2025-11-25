"""
Simple viewer app that just renders the sea floor.
This is a minimal test to verify floor rendering works before adding the fish.
"""
import os
import numpy as np
import fast_cd_pyb as fcd
import fast_cody as fc




def simple_floor_viewer():
    """
    Minimal viewer that just displays the sea floor mesh with texture.
    """
    print("=" * 60)
    print("SIMPLE FLOOR VIEWER")
    print("=" * 60)
    
    # Get shader paths
    vertex_shader_path = fc.get_shader("./vertex_shader_16.glsl")
    fragment_shader_path = fc.get_shader("./fragment_shader.glsl")
    
    if not os.path.exists(vertex_shader_path):
        raise FileNotFoundError(f"Vertex shader not found: {vertex_shader_path}")
    if not os.path.exists(fragment_shader_path):
        raise FileNotFoundError(f"Fragment shader not found: {fragment_shader_path}")
    
    print(f"Vertex shader: {vertex_shader_path}")
    print(f"Fragment shader: {fragment_shader_path}")
    
    # Create viewer (16 bones, 16 modes - required by shader)
    viewer_base = fcd.fast_cd_viewer_custom_shader(vertex_shader_path, fragment_shader_path, 16, 16)
    print("Viewer created successfully")
    
    # Load sea floor (quad mesh version)
    # The data directory is at fast_cody/data/ (one level up from src/fast_cody/)
    script_dir = os.path.dirname(os.path.abspath(__file__))  # src/fast_cody/apps/
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))  # fast_cody/
    data_dir = os.path.join(project_root, "data")
    floor_path = os.path.join(data_dir, "sea_floor.obj")
    floor_texture_path = os.path.join(data_dir, "sandtexture.jpg")

    print(f"\nLooking for floor mesh: {floor_path}")
    print(f"Floor exists: {os.path.exists(floor_path)}")
    print(f"Texture exists: {os.path.exists(floor_texture_path)}")

    floor_id = None
    floor_center = None
    p0_floor = None
    z0_floor = None

    if os.path.exists(floor_path):
        print(f"\nLoading floor mesh: {floor_path}")
        try:
            # Load OBJ with fcd.readOBJ_tex (handles quads and provides texture coords)
            print(f"  Loading OBJ with fcd.readOBJ_tex...")
            [V_floor, TC, N, F_floor, FTC, FN] = fcd.readOBJ_tex(floor_path)
            print(f"  Loaded mesh: {V_floor.shape[0]} vertices, {F_floor.shape[0]} faces")
            print(f"  Face shape: {F_floor.shape}")
            if TC is not None:
                print(f"  UV coordinates: {TC.shape[0]} texture coordinates")
            print(f"  FTC shape: {FTC.shape if FTC is not None else None}")
            print(f"  F_floor range: [{F_floor.min()}, {F_floor.max()}], V_floor has {V_floor.shape[0]} vertices")
            print(f"  FTC range: [{FTC.min()}, {FTC.max()}], TC has {TC.shape[0]} coords" if FTC is not None else "  No FTC")

            # Check if we have quads (4 vertices per face)
            if F_floor.shape[1] == 4:
                print(f"  Detected quad mesh - triangulating...")
                # Triangulate quads: split each quad [v0, v1, v2, v3] into two triangles
                # Triangle 1: [v0, v1, v2]
                # Triangle 2: [v0, v2, v3]
                num_quads = F_floor.shape[0]
                F_floor_tri = np.zeros((num_quads * 2, 3), dtype=np.int32)
                F_floor_tri[0::2, :] = F_floor[:, [0, 1, 2]]  # First triangle of each quad
                F_floor_tri[1::2, :] = F_floor[:, [0, 2, 3]]  # Second triangle of each quad
                F_floor = F_floor_tri

                # Do the same for texture coordinate indices if available
                if FTC is not None and FTC.shape[1] == 4:
                    FTC_tri = np.zeros((num_quads * 2, 3), dtype=np.int32)
                    FTC_tri[0::2, :] = FTC[:, [0, 1, 2]]  # First triangle texture coords
                    FTC_tri[1::2, :] = FTC[:, [0, 2, 3]]  # Second triangle texture coords
                    FTC = FTC_tri
                    print(f"  Triangulated: {F_floor.shape[0]} triangles from {num_quads} quads")
                    print(f"  Triangulated FTC: {FTC.shape}")
            else:
                print(f"  Mesh already triangulated (3 vertices per face)")

            # Check if texture is available
            use_texture = (os.path.exists(floor_texture_path) and
                          TC is not None and FTC is not None and
                          TC.shape[0] > 0 and FTC.shape[0] > 0)

            if use_texture:
                print(f"  Texture coords: {TC.shape[0]} coords, {FTC.shape[0]} face texture indices")
            else:
                print(f"  No texture coordinates found, will use solid color")
            
            # Scale and position the floor
            scale = 3.0
            V_floor_scaled = V_floor * scale
            V_floor_scaled[:, 1] -= 2.0  # Position below origin
            
            # Store floor center for debugging
            floor_center = np.mean(V_floor_scaled, axis=0)
            print(f"  Floor center: [{floor_center[0]:.2f}, {floor_center[1]:.2f}, {floor_center[2]:.2f}]")
            print(f"  Floor bounds: X=[{V_floor_scaled[:, 0].min():.2f}, {V_floor_scaled[:, 0].max():.2f}], "
                  f"Y=[{V_floor_scaled[:, 1].min():.2f}, {V_floor_scaled[:, 1].max():.2f}], "
                  f"Z=[{V_floor_scaled[:, 2].min():.2f}, {V_floor_scaled[:, 2].max():.2f}]")
            
            # Add floor mesh to viewer - use mesh slot 0 (like minimal_test_viewer)
            V_floor_scaled = np.ascontiguousarray(V_floor_scaled, dtype=np.float64)
            F_floor_contig = np.ascontiguousarray(F_floor, dtype=np.int32)

            # IMPORTANT: Use set_mesh with ID 0 directly (no add_mesh call)
            # The viewer expects mesh 0 to be the primary mesh
            floor_id = 0
            viewer_base.set_mesh(V_floor_scaled, F_floor_contig, floor_id)
            print(f"  Floor mesh geometry set (ID={floor_id})")

            # Apply texture first (before weights) - matching multi_fish order
            if use_texture and TC is not None and FTC is not None:
                try:
                    TC = np.ascontiguousarray(TC, dtype=np.float64)
                    FTC = np.ascontiguousarray(FTC, dtype=np.int32)
                    print(f"  Applying texture: {floor_texture_path}")
                    viewer_base.set_texture(floor_texture_path, TC, FTC, floor_id)
                    print(f"  ✓ Texture set")
                except Exception as e:
                    print(f"  ✗ Warning: Could not apply texture: {e}")
                    import traceback
                    traceback.print_exc()
                    print(f"           Using colored floor instead.")
                    use_texture = False

            # Set weights - bind all vertices to bone 0 with weight 1.0
            # The vertex shader REQUIRES non-zero weights or vertices collapse to origin
            num_bones = 16
            num_modes_floor = 16
            num_verts = V_floor_scaled.shape[0]

            Wp_floor = np.zeros((num_verts, num_bones), dtype=np.float64)
            Wp_floor[:, 0] = 1.0  # All vertices fully bound to bone 0 (identity)
            Ws_floor = np.zeros((num_verts, num_modes_floor), dtype=np.float64)

            viewer_base.set_weights(Wp_floor, Ws_floor, floor_id)
            print(f"  Set weights: all vertices bound to bone 0")

            # Set rendering options - AFTER texture and weights
            if use_texture:
                viewer_base.set_show_lines(False, floor_id)  # Disable wireframe
                viewer_base.set_face_based(False, floor_id)  # Use False for textured rendering
                print(f"  Set rendering options for textured floor")
            else:
                # Set a visible color for the floor (non-textured)
                floor_color = np.array([200, 180, 150]) / 255.0  # Sand/beige color
                viewer_base.set_color(floor_color, floor_id)
                viewer_base.set_face_based(True, floor_id)  # Double-sided rendering
                viewer_base.invert_normals(True, floor_id)
                viewer_base.set_show_lines(False, floor_id)  # Disable wireframe
                print(f"  Set rendering options for colored floor")

            # Initialize identity bone transforms for the floor (BEFORE launch)
            p0_floor = np.zeros((num_bones * 12, 1), dtype=np.float64)
            for bone_idx in range(num_bones):
                base_idx = bone_idx * 12
                p0_floor[base_idx + 0] = 1.0   # Identity matrix diagonal
                p0_floor[base_idx + 5] = 1.0
                p0_floor[base_idx + 10] = 1.0
            z0_floor = np.zeros((num_modes_floor * 12, 1), dtype=np.float64)

            viewer_base.set_bone_transforms(p0_floor, z0_floor, floor_id)
            print(f"  Initialized identity bone transforms for static floor")

            print("  Floor mesh loaded successfully!")
        except Exception as e:
            print(f"ERROR: Failed to load floor mesh: {e}")
            import traceback
            traceback.print_exc()
            return
    else:
        print(f"ERROR: Floor OBJ file not found: {floor_path}")
        return
    
    # Simple pre-draw callback to update the floor (even though it's static)
    frame_count = 0
    def pre_draw_callback():
        nonlocal floor_id, floor_center, frame_count, p0_floor, z0_floor
        frame_count += 1
        if floor_id is not None and p0_floor is not None:
            try:
                # Set bone transforms (even though static, shader might require them each frame)
                viewer_base.set_bone_transforms(p0_floor, z0_floor, floor_id)
                # Update the floor mesh (even though it's static, this ensures it renders)
                viewer_base.updateGL(floor_id)
                # Debug output on first few frames
                if frame_count == 1:
                    print(f"[DEBUG] Frame {frame_count}: Updated floor mesh (ID={floor_id})")
                    print(f"  Floor center: {floor_center}")
            except Exception as e:
                print(f"[ERROR] Failed to update floor mesh: {e}")
                import traceback
                traceback.print_exc()
        else:
            if frame_count <= 3:
                print(f"[WARNING] Frame {frame_count}: floor_id={floor_id}, p0_floor is None={p0_floor is None}")
    
    viewer_base.set_pre_draw_callback(pre_draw_callback)
    
    # Set camera to look at the floor
    # Position camera above and to the side of the floor to get a good view
    if floor_center is not None:
        # Camera positioned above and back from floor center
        # The floor is at Y=-2, so position camera higher
        camera_distance = 10.0  # Distance from floor
        camera_height = 5.0     # Height above floor
        camera_eye = floor_center + np.array([0.0, camera_height, -camera_distance], dtype=np.float64)
        camera_eye = camera_eye.reshape(1, 3)
        camera_center = floor_center.reshape(1, 3)
    else:
        camera_eye = np.array([[0.0, 3.0, -10.0]], dtype=np.float64)
        camera_center = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
    
    viewer_base.set_camera_eye(camera_eye)
    viewer_base.set_camera_center(camera_center)
    print(f"\nCamera set:")
    print(f"  Eye: {camera_eye.flatten()}")
    print(f"  Center: {camera_center.flatten()}")
    
    print("\n" + "=" * 60)
    print("Launching viewer...")
    print("=" * 60)
    
    # Launch viewer (max_fps=60, render=True)
    viewer_base.launch(60, True)
    
    print("Viewer closed.")


if __name__ == "__main__":
    simple_floor_viewer()

