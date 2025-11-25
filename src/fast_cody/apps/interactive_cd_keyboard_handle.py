#!/usr/bin/env python3
"""
Keyboard-controlled version of interactive fast CD simulation.

Controls:
  W/S: move forward/back
  A/D: move left/right
  Q/E: move up/down
  J/L: rotate left/right (yaw)

Continuously updates affine handle transformation to reproduce
the same deformation as the gizmo viewer, but via keyboard input.
"""

import os
import numpy as np
import fast_cd_pyb as fcd
import fast_cody as fc
import glfw


def rotation_matrix_yaw(yaw):
    """Return rotation matrix for small yaw (rotation about Y axis)."""
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, 0, s],
                     [0, 1, 0],
                     [-s, 0, c]], dtype=np.float32)


def interactive_cd_keyboard_handle(
    msh_file=None, V=None, T=None, Ws=None, l=None,
    mu=1e4, rho=1e3, num_modes=16, num_clusters=100,
    constraint_enforcement="optimal",
    cache_dir=None, results_dir=None, read_cache=False,
    texture_png=None, texture_obj=None
):
    """Run fast-CD simulation controlled via keyboard input."""

    # --- Load geometry ---
    if msh_file is not None:
        [V, F, T] = fcd.readMSH(msh_file)
    elif msh_file is None and (V is None and T is None):
        msh_file = fc.get_data("./cd_fish.msh")
        [V, F, T] = fcd.readMSH(msh_file)
    else:
        assert (V is not None and T is not None), "Must provide either msh_file or V and T"

    if texture_png is None or texture_obj is None:
        if msh_file == fc.get_data("./cd_fish.msh"):
            texture_png = fc.get_data("./cd_fish_tex.png")
            texture_obj = fc.get_data("./cd_fish_tex.obj")

    # --- Setup cache ---
    if cache_dir is None:
        cache_dir = "./cache/"
    os.makedirs(cache_dir, exist_ok=True)

    # --- Normalize geometry ---
    [V, so, to] = fcd.scale_and_center_geometry(V, 1, np.array([[0, 0, 0.]]))

    # --- Handle setup ---
    Wp = np.ones((V.shape[0], 1))
    J = fc.lbs_jacobian(V, Wp)

    if Ws is None or l is None:
        C = fc.complementary_constraint_matrix(V, T, J, dt=1e-3)
        C2 = fc.lbs_weight_space_constraint(V, C)
        [B, l, Ws] = fc.skinning_subspace(
            V, T, num_modes, num_clusters,
            C=C2, read_cache=read_cache, cache_dir=cache_dir,
            constraint_enforcement=constraint_enforcement
        )
    else:
        num_modes = Ws.shape[1]
        num_clusters = l.max() + 1

    sim = fc.fast_cd_sim(V, T, B, l, J, mu=mu, rho=rho, h=1e-2,
                         cache_dir=cache_dir, read_cache=read_cache)

    z0 = np.zeros((num_modes * 12, 1))
    T0 = np.identity(4).astype(np.float32, order="F")
    p0 = T0[0:3, :].reshape((12, 1))
    st = fc.fast_cd_state(z0, p0)

    # --- Viewer ---
    viewer = fc.viewers.interactive_handle_subspace_viewer(
        V, T, Wp, Ws, None,
        T0=T0,
        texture_png=texture_png, texture_obj=texture_obj,
        t0=to, s0=so, init_guizmo=False
    )

    translation_speed = 0.01
    rotation_speed = 0.02

    def pre_draw_callback():
        nonlocal T0, st

        # --- Poll keyboard each frame ---
        win = viewer.window
        move = np.zeros(3)

        if glfw.get_key(win, glfw.KEY_W) == glfw.PRESS:
            move += np.array([0, 0, -translation_speed])
        if glfw.get_key(win, glfw.KEY_S) == glfw.PRESS:
            move += np.array([0, 0, translation_speed])
        if glfw.get_key(win, glfw.KEY_A) == glfw.PRESS:
            move += np.array([-translation_speed, 0, 0])
        if glfw.get_key(win, glfw.KEY_D) == glfw.PRESS:
            move += np.array([translation_speed, 0, 0])
        if glfw.get_key(win, glfw.KEY_Q) == glfw.PRESS:
            move += np.array([0, translation_speed, 0])
        if glfw.get_key(win, glfw.KEY_E) == glfw.PRESS:
            move += np.array([0, -translation_speed, 0])

        if np.linalg.norm(move) > 0:
            T0[:3, 3] += move

        # --- Rotation ---
        if glfw.get_key(win, glfw.KEY_J) == glfw.PRESS:
            R = rotation_matrix_yaw(rotation_speed)
            T0[:3, :3] = R @ T0[:3, :3]
        if glfw.get_key(win, glfw.KEY_L) == glfw.PRESS:
            R = rotation_matrix_yaw(-rotation_speed)
            T0[:3, :3] = R @ T0[:3, :3]

        # --- Sim step ---
        p = T0[0:3, :].reshape((12, 1))
        z = sim.step(p, st)
        st.update(z, p)
        viewer.update_subspace_coefficients(z, p)
        viewer.T0 = T0

    # Replace default callback
    viewer.pre_draw_callback = pre_draw_callback
    viewer.launch()


if __name__ == "__main__":
    interactive_cd_keyboard_handle()
