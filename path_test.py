# %%
import numpy as np
import uaibot as ub
from uaibot.simobjects.curve import CurveSE3
from uaibot import CurveSE3


def circle_with_barrel_roll(
    radius=3.0, z0=-3.0, center_xy=(0.0, -700.0), n_points=400, n_rolls=1.0
):
    """
    Sample a circle in the world z = z0 plane, with a 360*n_rolls-degree
    barrel roll about the tangent axis over one lap. Returns an (n, 4, 4)
    array of homogeneous transforms in the world frame.
    """
    cx, cy = center_xy
    curve = np.zeros((n_points, 4, 4))
    for i in range(n_points):
        s = 2.0 * np.pi * i / n_points  # lap angle
        phi = n_rolls * s  # barrel-roll angle

        # Position
        p = np.array([cx + radius * np.cos(s), cy + radius * np.sin(s), z0])

        # Pre-roll frame: x = tangent, z = world up, y = z x x
        x_axis = np.array([-np.sin(s), np.cos(s), 0.0])
        z_axis = np.array([0.0, 0.0, 1.0])
        y_axis = np.cross(z_axis, x_axis)
        y_axis /= np.linalg.norm(y_axis)
        z_axis = np.cross(x_axis, y_axis)
        R0 = np.column_stack([x_axis, y_axis, z_axis])

        # Barrel roll: rotate around x_axis by phi
        c, sn = np.cos(phi), np.sin(phi)
        R_barrel = np.array([[1.0, 0.0, 0.0], [0.0, c, -sn], [0.0, sn, c]])
        R = R0 @ R_barrel
        # R = np.eye(3)

        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = p
        curve[i] = T
    return curve


MAX_TIME = 50.0
CURVE_X0, CURVE_Y0, CURVE_Z0 = -2.0, -2.0, -2.0
CURVE_POINTS = 5000
CURVE_RADIUS = 1.5
CURVE_N_ROLLS = 1.0

curve = circle_with_barrel_roll(
    radius=CURVE_RADIUS,
    z0=CURVE_Z0,
    center_xy=(CURVE_X0, CURVE_Y0),
    n_points=CURVE_POINTS,
    n_rolls=CURVE_N_ROLLS,
)

curve_ub = CurveSE3(name="curve", points=curve)
type(curve_ub)
sim = ub.Simulation(curve_ub)
sim.run_in_browser()

