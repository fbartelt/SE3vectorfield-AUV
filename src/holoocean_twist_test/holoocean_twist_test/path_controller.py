#!/usr/bin/env python3
"""
SE(3) vector-field path-following controller.

Subscribes:
    /holoocean/auv0/DynamicsSensorOdom  nav_msgs/Odometry  (pose in world)

Publishes:
    /controller/cmd_vel_world           geometry_msgs/TwistStamped  (world frame)
"""

import os
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry
from datetime import datetime

# Import your controller. Adjust the import path to match your package.
import uaibot as ub


# ----------------------------------------------------------------------
# Curve generation: circle of radius R with a full 360 deg barrel roll
# ----------------------------------------------------------------------
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
    omega = 2.0 * np.pi
    S_omega = np.array([[0.0, -omega, 0.0], [omega, 0.0, 0.0], [0.0, 0.0, 0.0]])
    dcurve = np.zeros((n_points, 4, 4))

    for i in range(n_points):
        s = 2.0 * np.pi * i / (n_points - 1)  # lap angle
        phi = n_rolls * s  # barrel-roll angle
        c, sn = np.cos(s), np.sin(s)
        # Position
        p = np.array([cx + radius * c, cy + radius * sn, z0])
        # dp/ds
        dp_ds = np.array([-omega * radius * sn, omega * radius * c, 0.0])

        # Pre-roll frame: x = tangent, z = world up, y = z x x
        x_axis = np.array([-np.sin(s), np.cos(s), 0.0])
        x_axis /= np.linalg.norm(x_axis)
        z_axis = np.array([0.0, 0.0, 1.0])
        y_axis = np.cross(z_axis, x_axis)
        y_axis /= np.linalg.norm(y_axis)
        z_axis = np.cross(x_axis, y_axis)
        R0 = np.column_stack([x_axis, y_axis, z_axis])

        R = R0

        # Barrel roll: rotate around x_axis by phi
        # phi = np.deg2rad(-30) * 0
        # c, sn = np.cos(phi), np.sin(phi)
        # R_barrel = np.array([[1.0, 0.0, 0.0], [0.0, c, -sn], [0.0, sn, c]])
        # R = R0 @ R_barrel
        # R = np.eye(3)

        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = p
        curve[i] = T
        dT = np.zeros((4, 4))
        dT[:3, :3] = S_omega @ R
        dT[:3,   3] = dp_ds
        dcurve[i] = dT
    return curve, dcurve


# ----------------------------------------------------------------------
def quat_to_rot(x, y, z, w):
    n = x * x + y * y + z * z + w * w
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    xx, yy, zz = x * x * s, y * y * s, z * z * s
    xy, xz, yz = x * y * s, x * z * s, y * z * s
    wx, wy, wz = w * x * s, w * y * s, w * z * s
    return np.array(
        [
            [1 - (yy + zz), xy - wz, xz + wy],
            [xy + wz, 1 - (xx + zz), yz - wx],
            [xz - wy, yz + wx, 1 - (xx + yy)],
        ]
    )


class PathController(Node):
    def __init__(self):
        super().__init__("path_controller")

        self.declare_parameter("agent_name", "auv0")
        self.declare_parameter("rate_hz", 20.0)
        self.declare_parameter("radius", 3.0)
        self.declare_parameter("z0", -3.0)
        self.declare_parameter("center_x", 0.0)
        self.declare_parameter("center_y", -700.0)
        self.declare_parameter("n_points", 5000)
        self.declare_parameter("n_rolls", 1.0)
        # Vector field gains - expose them so you can tune without rebuilding
        self.declare_parameter("kt1", 1.0)
        self.declare_parameter("kt2", 1.0)
        self.declare_parameter("kt3", 1.0)
        self.declare_parameter("kn1", 1.0)
        self.declare_parameter("kn2", 1.0)
        # Set this to False if the twist returned is already in the world frame
        self.declare_parameter("twist_is_body_frame", False)
        # Set this to True if the twist ordering is [omega; v]
        self.declare_parameter("angular_first", False)

        self.agent_name = self.get_parameter("agent_name").value
        rate = self.get_parameter("rate_hz").value
        self.twist_is_body = self.get_parameter("twist_is_body_frame").value
        self.angular_first = self.get_parameter("angular_first").value

        # Build curve
        curve_, curve_derivative_ = circle_with_barrel_roll(
            radius=float(self.get_parameter("radius").value),
            z0=float(self.get_parameter("z0").value),
            center_xy=(
                float(self.get_parameter("center_x").value),
                float(self.get_parameter("center_y").value),
            ),
            n_points=int(self.get_parameter("n_points").value),
            n_rolls=float(self.get_parameter("n_rolls").value),
        )
        self.curve = curve_
        self.curve_derivative = curve_derivative_

        self.get_logger().info(
            f"Curve built: {self.curve.shape[0]} samples, "
            f'radius={self.get_parameter("radius").value}, '
            f'rolls={self.get_parameter("n_rolls").value}'
        )

        self.htm = np.eye(4)
        self.have_pose = False
        self.start_time = self.get_clock().now()

        self.create_subscription(
            Odometry,
            f"/holoocean/{self.agent_name}/DynamicsSensorOdom",
            self.odom_cb,
            10,
        )
        self.pub = self.create_publisher(TwistStamped, "/controller/cmd_vel_world", 10)
        self.timer = self.create_timer(1.0 / rate, self.loop)
        # self.create_timer(1.0 / rate, self.loop)

        # --- data logging ---
        self.log_t = []
        self.log_htm = []
        self.log_closest = []
        self.log_dist = []
        self.log_idx = []
        self.run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.declare_parameter("max_sim_time", 0.0)  # seconds; 0 = no limit
        self.max_sim_time = float(self.get_parameter("max_sim_time").value)
        self.t0 = None

    def odom_cb(self, msg: Odometry):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        T = np.eye(4)
        T[:3, :3] = quat_to_rot(q.x, q.y, q.z, q.w)
        T[:3, 3] = [p.x, p.y, p.z]
        self.htm = T
        self.have_pose = True

    def _build_world_twist(self) -> np.ndarray:
        """Call the vector field and return [v; w] in the world frame."""
        qdot, dist, idx = ub.Robot.vector_field_SE3(
            self.htm,
            self.curve,
            kt1=float(self.get_parameter("kt1").value),
            kt2=float(self.get_parameter("kt2").value),
            kt3=float(self.get_parameter("kt3").value),
            kn1=float(self.get_parameter("kn1").value),
            kn2=float(self.get_parameter("kn2").value),
            curve_derivative=self.curve_derivative,
        )
        qdot = np.asarray(qdot).reshape(-1).copy()
        # qdot[3:6] = 0.0     # <-- TEMPORARY: kill angular component

        # --- log one sample ---
        t = self.get_clock().now().nanoseconds * 1e-9
        self.log_t.append(t)
        self.log_htm.append(self.htm.copy())
        self.log_closest.append(self.curve[idx].copy())
        self.log_dist.append(float(dist))
        self.log_idx.append(int(idx))

        self.get_logger().info(
            f"[VF] Cur htm: {self.htm}"
            f"[VF] dist={dist:.4f} idx={idx} xi={np.round(qdot, 4).tolist()}"
            f"[VF] Closest curve htm: {self.curve[idx]}"
        )

        # Reorder if the controller returns [omega; v]
        if self.angular_first:
            qdot = np.concatenate([qdot[3:6], qdot[0:3]])

        # Now qdot is [vx, vy, vz, wx, wy, wz]
        if not self.twist_is_body:
            return qdot

        # Controller returned body-frame twist -> convert to world for our
        # downstream adapter, which expects world.
        R = self.htm[:3, :3]
        v_world = R @ qdot[0:3]
        w_world = R @ qdot[3:6]
        return np.concatenate([v_world, w_world])

    def loop(self):
        if not self.have_pose:
            return
        t = self.get_clock().now().nanoseconds * 1e-9

        # --- auto-stop after max_sim_time ---
        if self.t0 is None:
            self.t0 = t
        elif self.max_sim_time > 0.0 and (t - self.t0) >= self.max_sim_time:
            self.get_logger().info(
                f"[RUN] reached max_sim_time = {self.max_sim_time} s, stopping"
            )
            # Stop the timer first so no more calls fire
            self.destroy_timer(self.timer)
            # Ask rclpy to exit the spin loop; main() will save logs
            rclpy.shutdown()
            return

        v_world = self._build_world_twist()

        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "world"
        vals = np.asarray(v_world).reshape(-1)
        msg.twist.linear.x = vals[0].item()
        msg.twist.linear.y = vals[1].item()
        msg.twist.linear.z = vals[2].item()
        msg.twist.angular.x = vals[3].item()
        msg.twist.angular.y = vals[4].item()
        msg.twist.angular.z = vals[5].item()
        # msg.twist.linear.x,  msg.twist.linear.y,  msg.twist.linear.z  = v_world[0:3]
        # msg.twist.angular.x, msg.twist.angular.y, msg.twist.angular.z = v_world[3:6]
        self.pub.publish(msg)

    def save_logs(self):
        out_dir = os.path.expanduser("~/holoocean_ws/logs")
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"path_track_{self.run_stamp}.npz")
        path = os.path.join(out_dir, "path_track.npz")
        np.savez(
            path,
            t=np.array(self.log_t),
            htm=np.array(self.log_htm),  # (N, 4, 4)
            closest=np.array(self.log_closest),  # (N, 4, 4)
            dist=np.array(self.log_dist),  # (N,)
            idx=np.array(self.log_idx),  # (N,)
        )
        self.get_logger().info(f"[LOG] saved {len(self.log_t)} samples to {path}")


def main(args=None):
    rclpy.init(args=args)
    node = PathController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.save_logs()
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
