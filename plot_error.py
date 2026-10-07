import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

data = np.load("/home/fbartelt/holoocean_ws/logs/path_track.npz")
pid = np.load("/home/fbartelt/holoocean_ws/logs/pid_trace.npz")

t = data["t"]  # (N,)
htm = data["htm"]  # (N, 4, 4)  — body pose in world
closest = data["closest"]  # (N, 4, 4)  — closest curve point
dist = data["dist"]  # (N,)       — SE(3) distance
closest_idxs = data["idx"]

# --- Unpack PID data ---
t_pid = pid["t"]
v_des = pid["v_des"]  # (N, 6)  body-frame desired  [u v w p q r]
v_cur = pid["v_cur"]  # (N, 6)  body-frame measured
err = pid["err"]  # (N, 6)
integral = pid["integral"]  # (N, 6)
derivative = pid["derivative"]  # (N, 6)
tau = pid["tau"]  # (N, 6)  wrench commanded
forces = pid["forces"]  # (N, 8)  thruster forces after clip

MAX_THRUST = 10.0 * 11.5 / 4.0  # BR_MAX_THRUST = 28.75 N


ori_errs = []
pos_errs = []
# Compute the distance, position error, and orientation error
for closest_point, state in zip(closest, htm):
    p_near = closest_point[:3, 3]
    ori_near = closest_point[:3, :3]
    p_curr = state[:3, 3]
    ori_curr = state[:3, :3]
    pos_errs.append(np.linalg.norm(p_near - p_curr) * 100)
    trace_ = np.trace(ori_near @ np.linalg.inv(ori_curr))
    acos = np.arccos((trace_ - 1) / 2)
    # checks if acos is nan
    if np.isnan(acos):
        acos = 0
    ori_errs.append(acos * 180 / np.pi)


def create_fig():
    # Create a figure with three plots, one above another. First the distance,
    # then position error, and the orientation error
    time_vec = t
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.02)
    fig.add_trace(
        go.Scatter(x=time_vec, y=dist, showlegend=False, line=dict(width=3)),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=time_vec, y=pos_errs, showlegend=False, line=dict(width=3)),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=time_vec, y=ori_errs, showlegend=False, line=dict(width=3)),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=time_vec, y=closest_idxs, showlegend=False, line=dict(width=3)),
        row=4,
        col=1,
    )
    fig.update_xaxes(
        title_text="Time (s)", gridcolor="gray", zerolinecolor="gray", row=3, col=1
    )
    fig.update_xaxes(
        title_text="", gridcolor="gray", zerolinecolor="gray", row=1, col=1
    )
    fig.update_xaxes(
        title_text="", gridcolor="gray", zerolinecolor="gray", row=2, col=1
    )
    fig.update_yaxes(
        title_text="Distance D",
        gridcolor="gray",
        zerolinecolor="gray",
        row=1,
        col=1,
        title_standoff=30,
    )
    fig.update_yaxes(
        title_text="Pos. error (cm)",
        gridcolor="gray",
        zerolinecolor="gray",
        row=2,
        col=1,
        title_standoff=30,
    )
    fig.update_yaxes(
        title_text="Ori. error (deg)",
        gridcolor="gray",
        zerolinecolor="gray",
        row=3,
        col=1,
        title_standoff=30,
    )
    fig.update_yaxes(
        title_text="Closest index",
        gridcolor="gray",
        zerolinecolor="gray",
        row=4,
        col=1,
        title_standoff=30,
    )


    # fig.update_layout(width=718.110, height=605.9155)
    fig.update_layout(width=1280, height=900)
    return fig


def create_pid_fig():
    # 6 rows: v_des vs v_cur, error, integral, derivative, tau, forces
    fig = make_subplots(
        rows=6,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=(
            "Velocity: desired vs current (body frame)",
            "Velocity error magnitudes (linear / angular)",
            "Integral term",
            "Derivative term",
            "Commanded wrench tau",
            "Thruster forces (N)",
        ),
    )

    # Rows 1–5 have 6 DOF each; use one trace per DOF with named colors
    dof_names = [
        "u (surge)",
        "v (sway)",
        "w (heave)",
        "p (roll)",
        "q (pitch)",
        "r (yaw)",
    ]

    # --- Row 1: v_des (solid) and v_cur (dashed) for all 6 DOF ---
    for i in range(6):
        fig.add_trace(
            go.Scatter(
                x=t_pid,
                y=v_des[:, i],
                name=f"des {dof_names[i]}",
                line=dict(width=1.5),
                legendgroup=f"dof{i}",
                legendgrouptitle_text=dof_names[i],
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=t_pid,
                y=v_cur[:, i],
                name=f"cur {dof_names[i]}",
                line=dict(width=1.5, dash="dot"),
                legendgroup=f"dof{i}",
            ),
            row=1,
            col=1,
        )

    # --- Row 2: error, one trace per DOF ---
    fig.add_trace(
        go.Scatter(
            x=t_pid,
            y=np.linalg.norm(err[:, :3], axis=1),
            name="||e<sub>pos</sub>||",
            line=dict(width=1.5),
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=t_pid,
            y=np.linalg.norm(err[:, 3:], axis=1),
            name="||e<sub>ori</sub>||",
            line=dict(width=1.5),
        ),
        row=2,
        col=1,
    )


    # for i in range(6):
    #     fig.add_trace(
    #         go.Scatter(
    #             x=t_pid,
    #             y=err[:, i],
    #             name=f"e {dof_names[i]}",
    #             line=dict(width=1.5),
    #             showlegend=False,
    #         ),
    #         row=2,
    #         col=1,
    #     )

    # --- Row 3: integral ---
    for i in range(6):
        fig.add_trace(
            go.Scatter(
                x=t_pid, y=integral[:, i], line=dict(width=1.5), showlegend=False
            ),
            row=3,
            col=1,
        )

    # --- Row 4: derivative ---
    for i in range(6):
        fig.add_trace(
            go.Scatter(
                x=t_pid, y=derivative[:, i], line=dict(width=1.5), showlegend=False
            ),
            row=4,
            col=1,
        )

    # --- Row 5: tau (only show Fx, Fy, Fz, Mz in legend; roll/pitch often noisy) ---
    for i in range(6):
        fig.add_trace(
            go.Scatter(
                x=t_pid,
                y=tau[:, i],
                name=f"tau {dof_names[i]}",
                line=dict(width=1.5),
                showlegend=(i in (0, 1, 2, 5)),
            ),
            row=5,
            col=1,
        )

    # --- Row 6: 8 thruster forces, plus saturation lines ---
    for i in range(8):
        fig.add_trace(
            go.Scatter(x=t_pid, y=forces[:, i], name=f"T{i}", line=dict(width=1.0)),
            row=6,
            col=1,
        )
    # Saturation bands
    fig.add_hline(
        y=MAX_THRUST,
        line=dict(color="black", width=1, dash="dash"),
        row=6,
        col=1,
        annotation_text=f"+{MAX_THRUST:.1f} N",
    )
    fig.add_hline(
        y=-MAX_THRUST,
        line=dict(color="black", width=1, dash="dash"),
        row=6,
        col=1,
        annotation_text=f"-{MAX_THRUST:.1f} N",
    )

    # --- Axis labels ---
    fig.update_xaxes(title_text="Time (s)", row=6, col=1)
    fig.update_yaxes(title_text="m/s | rad/s", row=1, col=1, title_standoff=30)
    fig.update_yaxes(title_text="m/s | rad/s", row=2, col=1, title_standoff=30)
    fig.update_yaxes(title_text="integral", row=3, col=1, title_standoff=30)
    fig.update_yaxes(title_text="derivative", row=4, col=1, title_standoff=30)
    fig.update_yaxes(title_text="N | N·m", row=5, col=1, title_standoff=30)
    fig.update_yaxes(title_text="N", row=6, col=1, title_standoff=30)

    # fig.update_layout(
    #     width=1100,
    #     height=1400,
    #     hovermode="x unified",
    #     legend=dict(groupclick="toggleitem"),
    # )

    fig.update_layout(width=1600, height=1400)
    # Apply grey grid to every subplot
    fig.update_xaxes(gridcolor="gray", zerolinecolor="gray")
    fig.update_yaxes(gridcolor="gray", zerolinecolor="gray")

    return fig


fig = create_fig()
fig.show()
create_pid_fig().show()
