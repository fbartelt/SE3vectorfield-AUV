# %%
import numpy as np
import uaibot as ub
from uaibot.simobjects.curve import CurveSE3
from uaibot.simobjects.curve import CurveSE3
import sys, os

sys.path.append("./src/holoocean_twist_test/")
from holoocean_twist_test.curves import build_curve  # adjust import path

CURVE_X0, CURVE_Y0, CURVE_Z0 = -2.0 * 0, -2.0 * 0, -4.0
CURVE_POINTS = 5000

# lissajous_3d
# circle
# ellipse
CURVE_TYPE = "lissajous_3d"

CURVE_RADIUS = 1.5 * 2 * 2
CURVE_N_ROLLS = 0.0

# Ellipse
ELLIPSE_A = 2.0 * 10
ELLIPSE_B = 1.0 * 10

# Lissajous
LISSAJOUS_A = 20.0
LISSAJOUS_B = 15.0
LISSAJOUS_C = 1.0

kn1 = 0.6
kn2 = 1.0
kt1 = 0.05
kt2 = 1.0
kt3 = kn2

kwargs = {
    "n_points": CURVE_POINTS,
    "z0": CURVE_Z0,
    "center_xy": (CURVE_X0, CURVE_Y0),
}

if CURVE_TYPE == "circle":
    kwargs["radius"] = CURVE_RADIUS
elif CURVE_TYPE == "ellipse":
    kwargs["a"] = ELLIPSE_A
    kwargs["b"] = ELLIPSE_B
elif CURVE_TYPE == "lissajous_3d":
    kwargs["A"] = LISSAJOUS_A
    kwargs["B"] = LISSAJOUS_B
    kwargs["C"] = LISSAJOUS_C

curve, curve_derivative = build_curve(CURVE_TYPE, **kwargs)

curve_ub = CurveSE3(name="curve", points=curve)
type(curve_ub)
sim = ub.Simulation(curve_ub)
sim.run_in_browser()
