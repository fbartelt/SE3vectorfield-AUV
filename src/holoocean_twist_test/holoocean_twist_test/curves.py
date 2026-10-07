import numpy as np

# ==========================================================================
# Generic frame construction and analytic derivative
# ==========================================================================


def _frame_and_derivative(p, dp_ds, dp_ds2):
    """
    Given p(s), dp/ds, d2p/ds2, return the tangent-aligned body frame
    R = [t, y, z] and its derivative dR/ds.

    Frame convention:
        t = p'/|p'|                       (forward)
        y = (k x t)/|k x t|, k = (0,0,1)  (horizontal left)
        z = t x y                         (body up)
    """
    v = float(np.linalg.norm(dp_ds))
    t = dp_ds / v
    k = np.array([0.0, 0.0, 1.0])

    # dt/ds
    t_dot_ddp = float(np.dot(t, dp_ds2))
    dt = (dp_ds2 - t * t_dot_ddp) / v

    # y
    w = np.cross(k, t)
    w_norm = float(np.linalg.norm(w))
    y = w / w_norm

    # dy/ds
    dw = np.cross(k, dt)
    dw_norm = float(np.dot(w, dw)) / w_norm
    dy = dw / w_norm - w * dw_norm / (w_norm**2)

    # z
    z = np.cross(t, y)
    dz = np.cross(dt, y) + np.cross(t, dy)

    R = np.column_stack([t, y, z])
    dR = np.column_stack([dt, dy, dz])
    return R, dR


def _htm_and_derivative(p, dp_ds, dp_ds2):
    R, dR = _frame_and_derivative(p, dp_ds, dp_ds2)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p
    dT = np.zeros((4, 4))
    dT[:3, :3] = dR
    dT[:3, 3] = dp_ds
    return T, dT


def _sample(p_func, dp_func, ddp_func, n_points):
    """
    Sample a closed curve given its parametric functions.
    s ranges over [0, 1); the closing sample is implicit via wraparound.
    """
    curve = np.zeros((n_points, 4, 4))
    dcurve = np.zeros((n_points, 4, 4))
    for i in range(n_points):
        s = i / n_points
        p = p_func(s)
        dp = dp_func(s)
        ddp = ddp_func(s)
        T, dT = _htm_and_derivative(p, dp, ddp)
        curve[i] = T
        dcurve[i] = dT
    return curve, dcurve


# ==========================================================================
# Curve builders
# ==========================================================================


def make_circle(radius=1.5, z0=-1.0, center_xy=(0.0, 0.0), n_points=5000):
    cx, cy = center_xy
    w = 2.0 * np.pi
    w2 = w * w

    def p(s):
        th = w * s
        return np.array([cx + radius * np.cos(th), cy + radius * np.sin(th), z0])

    def dp(s):
        th = w * s
        return np.array([-w * radius * np.sin(th), w * radius * np.cos(th), 0.0])

    def ddp(s):
        th = w * s
        return np.array([-w2 * radius * np.cos(th), -w2 * radius * np.sin(th), 0.0])

    return _sample(p, dp, ddp, n_points)


def make_ellipse(a=2.0, b=1.0, z0=-1.0, center_xy=(0.0, 0.0), n_points=5000):
    cx, cy = center_xy
    w = 2.0 * np.pi
    w2 = w * w

    def p(s):
        th = w * s
        return np.array([cx + a * np.cos(th), cy + b * np.sin(th), z0])

    def dp(s):
        th = w * s
        return np.array([-w * a * np.sin(th), w * b * np.cos(th), 0.0])

    def ddp(s):
        th = w * s
        return np.array([-w2 * a * np.cos(th), -w2 * b * np.sin(th), 0.0])

    return _sample(p, dp, ddp, n_points)


def make_lissajous_3d(
    A=2.0, B=1.5, C=0.5, z0=-1.0, center_xy=(0.0, 0.0), n_points=5000
):
    """
    Closed 3D curve: 2:1 frequency ratio between (x,y) and z.
    No self-intersections because the vertical oscillation breaks the
    2D projection's degeneracy.
    """
    cx, cy = center_xy
    w = 2.0 * np.pi
    w2 = 2.0 * w  # z has twice the angular frequency
    w2_sq = w2 * w2

    def p(s):
        th = w * s
        ph = w2 * s
        return np.array([cx + A * np.cos(th), cy + B * np.sin(th), z0 + C * np.sin(ph)])

    def dp(s):
        th = w * s
        ph = w2 * s
        return np.array([-w * A * np.sin(th), w * B * np.cos(th), w2 * C * np.cos(ph)])

    def ddp(s):
        th = w * s
        ph = w2 * s
        return np.array(
            [-w * w * A * np.cos(th), -w * w * B * np.sin(th), -w2_sq * C * np.sin(ph)]
        )

    return _sample(p, dp, ddp, n_points)


# ==========================================================================
# Registry
# ==========================================================================
CURVE_BUILDERS = {
    "circle": make_circle,
    "ellipse": make_ellipse,
    "lissajous_3d": make_lissajous_3d,
}


def build_curve(curve_type, **kwargs):
    if curve_type not in CURVE_BUILDERS:
        raise ValueError(
            f"Unknown curve type: {curve_type}. " f"Available: {list(CURVE_BUILDERS)}"
        )
    return CURVE_BUILDERS[curve_type](**kwargs)
