#!/usr/bin/env python3
"""
Baseline controller: Rezende's Euclidean guiding vector field applied
to the SE(3) curve embedded in R^6 as (x, y, z, roll, pitch, yaw).

Reference:
    A. M. C. Rezende, V. M. Goncalves, L. C. A. Pimenta,
    "Constructive Time-Varying Vector Fields for Robot Navigation,"
    IEEE Transactions on Robotics, 2021.

Subscribes:
    /holoocean/auv0/DynamicsSensorOdom   nav_msgs/Odometry

Publishes:
    /controller/cmd_vel_world            geometry_msgs/TwistStamped
"""

import os
from datetime import datetime

import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry

import uaibot as ub
from holoocean_twist_test.curves import build_curve


# ============================================================================
# Rotation helpers
# ============================================================================
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


def rot_to_euler_zyx(R):
    """
    ZYX intrinsic Euler angles from a rotation matrix.
    Returns (roll, pitch, yaw) in radians.
    """
    # Orthogonalize first (numerical drift tolerance)
    U, _, Vt = np.linalg.svd(R)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt

    sp = -R[2, 0]
    sp = np.clip(sp, -1.0, 1.0)
    pitch = np.arcsin(sp)
    cp = np.cos(pitch)

    if abs(cp) > 1e-6:
        roll = np.arctan2(R[2, 1], R[2, 2])
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        # Gimbal lock
        roll = 0.0
        yaw = np.arctan2(-R[0, 1], R[1, 1])

    return np.array([roll, pitch, yaw])


def euler_rates_to_omega_world(roll, pitch, yaw, r_dot, p_dot, y_dot):
    """
    Convert ZYX Euler-rate triple to world-frame angular velocity.
    omega_world = E(roll, pitch, yaw) @ [r_dot, p_dot, y_dot]
    """
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)

    omega_x = cy * cp * r_dot - sy * p_dot
    omega_y = sy * cp * r_dot + cy * p_dot
    omega_z = -sp * r_dot + y_dot
    return np.array([omega_x, omega_y, omega_z])


# ============================================================================
# Controller
# ============================================================================
class BaselineController(Node):
    def __init__(self):
        super().__init__("baseline_controller")

        # --- params (mirror path_controller) ---
        self.declare_parameter("agent_name", "auv0")
        self.declare_parameter("rate_hz", 20.0)
        self.declare_parameter("curve_type", "circle")
        self.declare_parameter("radius", 1.5)
        self.declare_parameter("a", 2.0)
        self.declare_parameter("b", 1.0)
        self.declare_parameter("A", 2.0)
        self.declare_parameter("B", 1.5)
        self.declare_parameter("C", 0.5)
        self.declare_parameter("z0", -1.0)
        self.declare_parameter("center_x", 0.0)
        self.declare_parameter("center_y", 0.0)
        self.declare_parameter("n_points", 5000)
        self.declare_parameter("decoupled", False)
        # Rezende VF params
        self.declare_parameter("alpha", 1.0)
        self.declare_parameter("const_vel", 0.5)
        self.declare_parameter("gamma", 1.0)
        # auto-stop
        self.declare_parameter("max_sim_time", 0.0)

        self.agent_name = self.get_parameter("agent_name").value
        rate = float(self.get_parameter("rate_hz").value)
        self.max_sim_time = float(self.get_parameter("max_sim_time").value)
        self.t0 = None

        # --- build curve ---
        curve_type = self.get_parameter("curve_type").value
        self.curve_type = curve_type

        kwargs = {
            "n_points": int(self.get_parameter("n_points").value),
            "z0": float(self.get_parameter("z0").value),
            "center_xy": (
                float(self.get_parameter("center_x").value),
                float(self.get_parameter("center_y").value),
            ),
        }
        if curve_type == "circle":
            kwargs["radius"] = float(self.get_parameter("radius").value)
        elif curve_type == "ellipse":
            kwargs["a"] = float(self.get_parameter("a").value)
            kwargs["b"] = float(self.get_parameter("b").value)
        elif curve_type == "lissajous_3d":
            kwargs["A"] = float(self.get_parameter("A").value)
            kwargs["B"] = float(self.get_parameter("B").value)
            kwargs["C"] = float(self.get_parameter("C").value)

        curve_se3, _ = build_curve(curve_type, **kwargs)
        N = curve_se3.shape[0]

        positions = curve_se3[:, :3, 3].copy()
        eulers = np.zeros((N, 3))
        for i in range(N):
            eulers[i] = rot_to_euler_zyx(curve_se3[i, :3, :3])

        self.curve_se3 = curve_se3
        self.curve_pos = positions
        self.curve_euler = eulers
        # 6D embedding used for the VF query
        self.curve_r6 = np.column_stack([positions, eulers])

        self.get_logger().info(
            f"Baseline: curve={curve_type}, N={N}, "
            f"euler ranges (rad): "
            f"roll=[{eulers[:,0].min():.2f},{eulers[:,0].max():.2f}] "
            f"pitch=[{eulers[:,1].min():.2f},{eulers[:,1].max():.2f}] "
            f"yaw=[{eulers[:,2].min():.2f},{eulers[:,2].max():.2f}]"
        )

        # --- state ---
        self.htm = np.eye(4)
        self.have_pose = False

        # --- ROS ---
        self.create_subscription(
            Odometry,
            f"/holoocean/{self.agent_name}/DynamicsSensorOdom",
            self.odom_cb,
            10,
        )
        self.pub = self.create_publisher(TwistStamped, "/controller/cmd_vel_world", 10)
        self.timer = self.create_timer(1.0 / rate, self.loop)

        # --- logging ---
        self.log_t = []
        self.log_htm = []
        self.log_closest = []
        self.log_dist = []
        self.log_idx = []
        self.run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def odom_cb(self, msg):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        T = np.eye(4)
        T[:3, :3] = quat_to_rot(q.x, q.y, q.z, q.w)
        T[:3, 3] = [p.x, p.y, p.z]
        self.htm = T
        self.have_pose = True

    def _query_point_r6(self):
        """
        Assemble current pose into a 6D query point. Euler angles are
        aligned to the branch of the nearest curve sample (in position)
        so the R^6 nearest-point search sees consistent values.
        """
        p = self.htm[:3, 3]
        euler = rot_to_euler_zyx(self.htm[:3, :3])

        # Position-nearest sample for branch alignment
        d = np.linalg.norm(self.curve_pos - p, axis=1)
        idx_pos = int(np.argmin(d))
        euler_ref = self.curve_euler[idx_pos]

        euler_aligned = np.empty(3)
        for k in range(3):
            diff = euler[k] - euler_ref[k]
            diff = (diff + np.pi) % (2.0 * np.pi) - np.pi
            euler_aligned[k] = euler_ref[k] + diff

        return np.concatenate([p, euler_aligned])

    def _build_world_twist(self):
        q6 = self._query_point_r6()

        vf_out, dist, idx = ub.Robot.vector_field(
            q6,
            self.curve_r6,
            alpha=float(self.get_parameter("alpha").value),
            const_vel=float(self.get_parameter("const_vel").value),
            is_closed=True,
            gamma=float(self.get_parameter("gamma").value),
        )
        vf_out = np.asarray(vf_out).reshape(-1)

        # first 3 -> world-frame linear velocity
        v_world = vf_out[0:3]
        # last 3 -> Euler rates (roll_dot, pitch_dot, yaw_dot)
        r_dot, p_dot, y_dot = vf_out[3], vf_out[4], vf_out[5]

        # Convert to world angular velocity
        roll, pitch, yaw = q6[3], q6[4], q6[5]
        omega_world = euler_rates_to_omega_world(roll, pitch, yaw, r_dot, p_dot, y_dot)

        # --- log ---
        t = self.get_clock().now().nanoseconds * 1e-9
        self.log_t.append(t)
        self.log_htm.append(self.htm.copy())
        self.log_closest.append(self.curve_se3[idx].copy())
        self.log_dist.append(float(dist))
        self.log_idx.append(int(idx))

        return np.concatenate([v_world, omega_world])

    def loop(self):
        if not self.have_pose:
            return
        t = self.get_clock().now().nanoseconds * 1e-9

        if self.t0 is None:
            self.t0 = t
        elif self.max_sim_time > 0.0 and (t - self.t0) >= self.max_sim_time:
            self.get_logger().info(
                f"[RUN] reached max_sim_time = {self.max_sim_time} s"
            )
            self.destroy_timer(self.timer)
            rclpy.shutdown()
            return

        twist = self._build_world_twist()

        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "world"
        msg.twist.linear.x = float(twist[0])
        msg.twist.linear.y = float(twist[1])
        msg.twist.linear.z = float(twist[2])
        msg.twist.angular.x = float(twist[3])
        msg.twist.angular.y = float(twist[4])
        msg.twist.angular.z = float(twist[5])
        self.pub.publish(msg)

    def save_logs(self):
        out_dir = os.path.expanduser("~/holoocean_ws/logs")
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"baseline_track_{self.run_stamp}.npz")
        path = os.path.join(out_dir, f"baseline_track_{self.curve_type}.npz")
        np.savez(
            path,
            t=np.array(self.log_t),
            htm=np.array(self.log_htm),
            closest=np.array(self.log_closest),
            dist=np.array(self.log_dist),
            idx=np.array(self.log_idx),
        )
        self.get_logger().info(f"[LOG] saved {len(self.log_t)} samples to {path}")


def main(args=None):
    rclpy.init(args=args)
    node = BaselineController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.save_logs()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
