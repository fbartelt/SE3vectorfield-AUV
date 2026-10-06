"""
Rebuild the BlueROV2 thruster configuration matrix T from the HoloOcean source.

Sources:
    ~/holoocean/engine/Source/Holodeck/Agents/Public/BlueROV2.h
    ~/holoocean/engine/Source/Holodeck/Agents/Private/BlueROV2.cpp

Frames:
    UE body (left-handed):  X forward, Y starboard, Z up
    ROS body (right-handed): X forward, Y port,      Z up
    Mapping: (x,y,z)_ROS = (x, -y, z)_UE   and lengths converted cm -> m.

Output:
    T is 6x8. Row order [Fx, Fy, Fz, Mx, My, Mz]. Column i is thruster i.
    T[:, i] = [ d_i ; p_i x d_i ]
"""

import numpy as np


# ---------------------------------------------------------------------------
# 1. Raw thruster positions in the UE mesh frame (cm, LHS) — straight from
#    ABlueROV2::thrusterLocations in BlueROV2.h.
# ---------------------------------------------------------------------------
raw_positions_UE_cm = np.array([
    [ 12.00,  21.81,  7.09],  # 0: vertical front starboard
    [ 12.00, -21.81,  7.09],  # 1: vertical front port
    [-12.00, -21.81,  7.09],  # 2: vertical back port
    [-12.00,  21.81,  7.09],  # 3: vertical back starboard
    [ 15.62,   9.88, -1.00],  # 4: angled front starboard
    [ 15.62,  -9.88, -1.00],  # 5: angled front port
    [-15.62,  -9.88, -1.00],  # 6: angled back port
    [-15.62,   9.88, -1.00],  # 7: angled back starboard
])

# ---------------------------------------------------------------------------
# 2. Center of mass — exactly as ABlueROV2::InitializeAgent computes it for
#    Perfect = true.
# ---------------------------------------------------------------------------
center_mass_UE_cm = (raw_positions_UE_cm[0] + raw_positions_UE_cm[2]) / 2.0
center_mass_UE_cm[2] = raw_positions_UE_cm[7][2]

# ---------------------------------------------------------------------------
# 3. Thruster positions relative to COM, in UE body frame (cm).
#    This matches the for-loop in InitializeAgent.
# ---------------------------------------------------------------------------
positions_rel_UE_cm = raw_positions_UE_cm - center_mass_UE_cm

# ---------------------------------------------------------------------------
# 4. Convert positions to ROS body frame: y-flip and cm -> m.
# ---------------------------------------------------------------------------
positions_ROS_m = positions_rel_UE_cm * np.array([1.0, -1.0, 1.0]) / 100.0

# ---------------------------------------------------------------------------
# 5. Thruster directions in the ROS body frame.
#
#    In ABlueROV2::ApplyThrusters, the local force is written in a "client"
#    frame before ConvertLinearVector(., ClientToUE) is applied. The client
#    frame is the ROS/right-handed body frame; ClientToUE then maps into UE
#    by flipping the y-component. So the ROS-frame direction is exactly what
#    is written in the C++ assignment:
#
#        i < 4                 : (0, 0, 1)
#        i % 2 == 0  (i=4, 6)  : ( 1/sqrt(2),  1/sqrt(2), 0)
#        i % 2 == 1  (i=5, 7)  : ( 1/sqrt(2), -1/sqrt(2), 0)
# ---------------------------------------------------------------------------
s2 = 1.0 / np.sqrt(2.0)
dirs_ROS = np.array([
    [0.0,  0.0, 1.0],   # 0
    [0.0,  0.0, 1.0],   # 1
    [0.0,  0.0, 1.0],   # 2
    [0.0,  0.0, 1.0],   # 3
    [s2,   s2, 0.0],    # 4
    [s2,  -s2, 0.0],    # 5
    [s2,   s2, 0.0],    # 6
    [s2,  -s2, 0.0],    # 7
])

# ---------------------------------------------------------------------------
# 6. Assemble the configuration matrix T (6 x 8).
# ---------------------------------------------------------------------------
T = np.zeros((6, 8))
for i in range(8):
    F = dirs_ROS[i]
    M = np.cross(positions_ROS_m[i], F)
    T[0:3, i] = F
    T[3:6, i] = M


if __name__ == "__main__":
    np.set_printoptions(precision=8, suppress=True, linewidth=160)

    print("=== Thruster geometry in ROS body frame (m) ===")
    for i in range(8):
        p = positions_ROS_m[i]
        d = dirs_ROS[i]
        print(f"  {i}:  p = [{p[0]: .4f}, {p[1]: .4f}, {p[2]: .4f}]   "
              f"d = [{d[0]: .4f}, {d[1]: .4f}, {d[2]: .4f}]")

    print("\n=== Configuration matrix T (6x8) ===")
    print("rows: [Fx, Fy, Fz, Mx, My, Mz]")
    print(T)

    print("\nrank(T) =", np.linalg.matrix_rank(T))
    print("cond(T) =", f"{np.linalg.cond(T):.2f}")
