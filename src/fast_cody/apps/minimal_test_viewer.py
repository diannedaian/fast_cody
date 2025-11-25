"""
Minimal test viewer - just render a simple colored cube to verify the renderer works.
"""
import os
import numpy as np
import fast_cd_pyb as fcd
import fast_cody as fc


def minimal_test_viewer():
    """
    Simplest possible test - render a colored cube.
    """
    print("=" * 60)
    print("MINIMAL TEST VIEWER - Rendering a simple cube")
    print("=" * 60)

    # Get shader paths
    vertex_shader_path = fc.get_shader("./vertex_shader_16.glsl")
    fragment_shader_path = fc.get_shader("./fragment_shader.glsl")

    print(f"Vertex shader: {vertex_shader_path}")
    print(f"Fragment shader: {fragment_shader_path}")

    # Create viewer (16 bones, 16 modes - required by shader)
    viewer_base = fcd.fast_cd_viewer_custom_shader(vertex_shader_path, fragment_shader_path, 16, 16)
    print("Viewer created successfully")

    # Create a simple cube mesh
    # Vertices of a unit cube centered at origin
    V_cube = np.array([
        [-1, -1, -1],  # 0
        [ 1, -1, -1],  # 1
        [ 1,  1, -1],  # 2
        [-1,  1, -1],  # 3
        [-1, -1,  1],  # 4
        [ 1, -1,  1],  # 5
        [ 1,  1,  1],  # 6
        [-1,  1,  1],  # 7
    ], dtype=np.float64)

    # Faces of the cube (2 triangles per face)
    F_cube = np.array([
        # Front face
        [0, 1, 2], [0, 2, 3],
        # Back face
        [4, 6, 5], [4, 7, 6],
        # Left face
        [0, 3, 7], [0, 7, 4],
        # Right face
        [1, 5, 6], [1, 6, 2],
        # Top face
        [3, 2, 6], [3, 6, 7],
        # Bottom face
        [0, 4, 5], [0, 5, 1],
    ], dtype=np.int32)

    print(f"\nCube mesh: {V_cube.shape[0]} vertices, {F_cube.shape[0]} faces")
    print(f"Cube bounds: X=[{V_cube[:, 0].min():.2f}, {V_cube[:, 0].max():.2f}], "
          f"Y=[{V_cube[:, 1].min():.2f}, {V_cube[:, 1].max():.2f}], "
          f"Z=[{V_cube[:, 2].min():.2f}, {V_cube[:, 2].max():.2f}]")

    # Add cube mesh to viewer
    V_cube_contig = np.ascontiguousarray(V_cube, dtype=np.float64)
    F_cube_contig = np.ascontiguousarray(F_cube, dtype=np.int32)
    cube_id = viewer_base.add_mesh(V_cube_contig, F_cube_contig)
    print(f"Cube mesh ID: {cube_id}")

    # Set weights - bind all vertices to bone 0
    num_bones = 16
    num_modes = 16
    num_verts = V_cube.shape[0]

    Wp_cube = np.zeros((num_verts, num_bones), dtype=np.float64)
    Wp_cube[:, 0] = 1.0  # All vertices bound to bone 0
    Ws_cube = np.zeros((num_verts, num_modes), dtype=np.float64)

    viewer_base.set_weights(Wp_cube, Ws_cube, cube_id)
    print(f"Set weights: all vertices bound to bone 0")

    # Initialize identity bone transforms
    p0_cube = np.zeros((num_bones * 12, 1), dtype=np.float64)
    for bone_idx in range(num_bones):
        base_idx = bone_idx * 12
        p0_cube[base_idx + 0] = 1.0   # Identity matrix diagonal
        p0_cube[base_idx + 5] = 1.0
        p0_cube[base_idx + 10] = 1.0
    z0_cube = np.zeros((num_modes * 12, 1), dtype=np.float64)

    viewer_base.set_bone_transforms(p0_cube, z0_cube, cube_id)
    print(f"Initialized identity bone transforms")

    # Set a bright color for the cube (bright red)
    cube_color = np.array([1.0, 0.0, 0.0])  # Bright red
    viewer_base.set_color(cube_color, cube_id)
    print(f"Set cube color to bright red")

    # Set rendering options
    viewer_base.set_face_based(True, cube_id)
    viewer_base.invert_normals(True, cube_id)
    viewer_base.set_show_lines(True, cube_id)  # Show wireframe to make it more visible
    print(f"Set rendering options (face_based=True, invert_normals=True, show_lines=True)")

    # Simple pre-draw callback
    frame_count = 0
    def pre_draw_callback():
        nonlocal frame_count, cube_id, p0_cube, z0_cube
        frame_count += 1
        if cube_id is not None:
            try:
                viewer_base.set_bone_transforms(p0_cube, z0_cube, cube_id)
                viewer_base.updateGL(cube_id)
                if frame_count <= 3:
                    print(f"[DEBUG] Frame {frame_count}: Updated cube mesh (ID={cube_id})")
            except Exception as e:
                print(f"[ERROR] Failed to update cube mesh: {e}")
                import traceback
                traceback.print_exc()

    viewer_base.set_pre_draw_callback(pre_draw_callback)

    # Set camera to look at the cube
    camera_eye = np.array([[0.0, 0.0, -5.0]], dtype=np.float64)  # Camera 5 units back
    camera_center = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)  # Look at origin

    viewer_base.set_camera_eye(camera_eye)
    viewer_base.set_camera_center(camera_center)
    print(f"\nCamera set:")
    print(f"  Eye: {camera_eye.flatten()}")
    print(f"  Center: {camera_center.flatten()}")

    print("\n" + "=" * 60)
    print("Launching viewer...")
    print("If you see a RED CUBE, the renderer is working!")
    print("=" * 60)

    # Launch viewer
    viewer_base.launch(60, True)

    print("Viewer closed.")


if __name__ == "__main__":
    minimal_test_viewer()