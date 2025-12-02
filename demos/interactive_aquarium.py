import os
import numpy as np
import scipy as sp
import igl

import fast_cd_pyb as fcd
import fast_cody as fc


def interactive_aquarium():
    """
    Interactive aquarium with a single fish that can be manipulated using an affine handle.
    The fish is centered in the scene and interactable via Guizmo.
    """

    # Load mesh from data/cd_fish/cd_fish.msh
    msh_file = fc.get_data("./cd_fish.msh")
    [V, F, T] = fcd.readMSH(msh_file)

    # Load texture from data/cd_fish/cd_fish_tex.png
    texture_png = fc.get_data("./cd_fish_tex.png")
    texture_obj = fc.get_data("./cd_fish_tex.obj")

    # Setup cache directory
    cache_dir = "./cache/"
    os.makedirs(cache_dir, exist_ok=True)

    # Scale and center geometry to unit height and about origin
    # This centers the fish in the scene
    [V, so, to] = fcd.scale_and_center_geometry(V, 1, np.array([[0, 0, 0.]]))

    # Create single affine handle (all vertices weighted equally)
    Wp = np.ones((V.shape[0], 1))  # Single handle skinning weight
    J = fc.lbs_jacobian(V, Wp)

    # Compute secondary motion weights and clusters
    num_modes = 16
    num_clusters = 100
    constraint_enforcement = "optimal"
    read_cache = False

    C = fc.complementary_constraint_matrix(V, T, J, dt=1e-3)
    C2 = fc.lbs_weight_space_constraint(V, C)
    [B, l, Ws] = fc.skinning_subspace(V, T, num_modes, num_clusters, C=C2,
                                     read_cache=read_cache,
                                     cache_dir=cache_dir,
                                     constraint_enforcement=constraint_enforcement)

    # Create fast CD simulation
    mu = 1e4  # Lame parameter
    rho = 1e3  # Density
    sim = fc.fast_cd_sim(V, T, B, l, J, mu=mu, rho=rho, h=1e-2,
                        cache_dir=cache_dir, read_cache=read_cache)

    # Initialize simulation state
    # z0: secondary motion coefficients (start at zero)
    # p0: primary handle transform (start at identity)
    z0 = np.zeros((num_modes * 12, 1))
    T0 = np.identity(4).astype(dtype=np.float32, order="F")
    p0 = T0[0:3, :].reshape((12, 1))
    st = fc.fast_cd_state(z0, p0)

    # Animation step counter
    step = 0

    # Pre-draw callback - matches interactive_cd_affine_handle exactly
    def pre_draw_callback():
        """Called before each frame is drawn"""
        nonlocal J, B, T0, sim, st, step

        # Get the current fish transform from the guizmo/user input
        # viewer.T0 is updated by the guizmo when the user manipulates it
        p = viewer.T0[0:3, :].reshape((12, 1))

        # Step the simulation forward
        z = sim.step(p, st)
        st.update(z, p)

        # Update the viewer with new coefficients
        viewer.update_subspace_coefficients(z, p)

        step += 1

    # Create interactive viewer with texture
    # This centers the fish and makes it interactable via Guizmo
    # Matches the exact call signature from interactive_cd_affine_handle
    viewer = fc.viewers.interactive_handle_subspace_viewer(
        V, T, Wp, Ws, pre_draw_callback, T0=T0,
        texture_png=texture_png, texture_obj=texture_obj,
        t0=to, s0=so, init_guizmo=True
    )

    # Set background color to ocean blue
    background_color = np.array([95, 191, 210]) / 255.0
    viewer.viewer.set_background_color(background_color)

    print("=" * 60)
    print("Interactive Aquarium")
    print("=" * 60)
    print("  Controls:")
    print("    - Use Guizmo to move, rotate, and scale the fish")
    print("    - g: Toggle Guizmo transform mode (translate/rotate/scale)")
    print("=" * 60)

    # Launch the viewer
    viewer.launch()


if __name__ == "__main__":
    interactive_aquarium()
