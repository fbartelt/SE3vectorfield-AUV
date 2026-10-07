#!/usr/bin/env python3
"""
Velocity adapter: spatial (world-frame) Twist -> thruster commands.

The upstream controller publishes a spatial twist xi_s = (v_s, omega_s),
i.e. the right-invariant / spatial Lie algebra element satisfying
    xi_s = Hdot * H^{-1}
in the world frame. The dynamics below operate in body frame, so we apply
the SE(3) adjoint to convert:

    omega_b = R^T * omega_s
    v_b     = R^T * (v_s + omega_s x p)

where H = [[R, p], [0, 1]] is the current body pose in the world frame.

Subscribes:
    /controller/cmd_vel_world    geometry_msgs/TwistStamped   (spatial, world)
    /cmd_vel                     geometry_msgs/Twist          (optional, body)
    <ns>/auv0/DynamicsSensorOdom nav_msgs/Odometry
    <ns>/auv0/DVLSensorVelocity  geometry_msgs/TwistWithCovarianceStamped
    <ns>/auv0/IMUSensor          sensor_msgs/Imu

Publishes:
    <ns>/command/agent           holoocean_interfaces/AgentCommand
"""

import os
from datetime import datetime
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, TwistStamped, TwistWithCovarianceStamped
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
from holoocean_interfaces.msg import AgentCommand
from holoocean_twist_test.thruster_model import T
from scipy.linalg import block_diag

# # --- 6x8 thruster configuration matrix T, ROS FLU body frame ---
# T = np.array(
#     [
#         [0.0, 0.0, 0.0, 0.0, 0.70710678, 0.70710678, 0.70710678, 0.70710678],
#         [0.0, 0.0, 0.0, 0.0, -0.70710678, 0.70710678, -0.70710678, 0.70710678],
#         [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
#         [-0.221, 0.221, 0.221, -0.221, 0.0, 0.0, 0.0, 0.0],
#         [-0.248, -0.248, 0.248, 0.248, 0.0, 0.0, 0.0, 0.0],
#         [0.0, 0.0, 0.0, 0.0, 0.0297, -0.0297, -0.0297, 0.0297],
#     ]
# )
T_PINV = np.linalg.pinv(T)
MAX_THRUST = 77.55  # N per thruster


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


class VelocityAdapter(Node):
    def __init__(self):
        super().__init__("twist_to_agent")

        self.declare_parameter("agent_name", "auv0")
        self.declare_parameter("control_rate_hz", 50.0)
        self.declare_parameter("debug_log", True)
        self.declare_parameter("max_v_lin", 2.0)  # m/s
        self.declare_parameter("max_v_ang", 3.0)  # rad/s

        self.agent_name = self.get_parameter("agent_name").value
        rate = self.get_parameter("control_rate_hz").value
        self.debug_log = self.get_parameter("debug_log").value
        self.max_v_lin = float(self.get_parameter("max_v_lin").value)
        self.max_v_ang = float(self.get_parameter("max_v_ang").value)

        # PID gains (body frame): [u, v, w, p, q, r]
        kp_p, kp_w = 40.0, 4.0 * 0.5
        self.kp = 1e-1 * np.array([kp_p, kp_p, kp_p, kp_w, kp_w, kp_w])
        ki_p, ki_w = 5.0, 0.5 * 0.5
        self.ki = 1e-1 * np.array([ki_p, ki_p, ki_p, ki_w, ki_w, ki_w])
        kd_p, kd_w = 2.0, 0.1 * 0.5
        self.kd = 1e-1 * np.array([kd_p, kd_p, kd_p, kd_w, kd_w, kd_w])
        self.i_clamp = np.array([5.0, 5.0, 5.0, 1.0, 1.0, 1.0])

        self.integral = np.zeros(6)
        self.prev_error = np.zeros(6)
        self.current_velocity = np.zeros(6)
        self.have_first_error = False  # <-- add this
        self.d_filtered = np.zeros(6)  # <-- and this, if you added D-filtering
        self.v_des_smoothed = np.zeros(6)  # <-- and this, if you added ramp smoothing

        # Command sources. Body-frame is fallback / manual. World (spatial) is primary.
        self.cmd_world = np.zeros(6)  # [v_s; w_s]  (spatial twist)
        self.cmd_body = np.zeros(6)  # [v_b; w_b]  (body twist)
        self.have_world_cmd = False
        self.have_body_cmd = False

        # Pose from odometry
        self.R_wb = np.eye(3)  # body -> world rotation
        self.p_world = np.zeros(3)  # body origin in world frame
        self.have_odom = False
        self.got_dvl = False
        self.got_imu = False

        self.last_time = None
        self.last_log_time = None

        # time / run control
        self.declare_parameter("max_sim_time", 0.0)  # 0 = no limit
        self.max_sim_time = float(self.get_parameter("max_sim_time").value)
        self.t0 = None
        self.run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # --- data logging ---
        self.log_t = []
        self.log_v_des = []
        self.log_v_cur = []
        self.log_err = []
        self.log_integral = []
        self.log_derivative = []
        self.log_tau = []
        self.log_forces = []

        # --- subscriptions ---
        self.create_subscription(
            TwistStamped, "/controller/cmd_vel_world", self.world_cmd_cb, 10
        )
        self.create_subscription(Twist, "/cmd_vel", self.body_cmd_cb, 10)
        self.create_subscription(
            Odometry, f"{self.agent_name}/DynamicsSensorOdom", self.odom_cb, 10
        )
        self.create_subscription(
            TwistWithCovarianceStamped,
            f"{self.agent_name}/DVLSensorVelocity",
            self.dvl_cb,
            10,
        )
        self.create_subscription(Imu, f"{self.agent_name}/IMUSensor", self.imu_cb, 10)

        self.pub = self.create_publisher(AgentCommand, "command/agent", 10)
        self.timer = self.create_timer(1.0 / rate, self.control_loop)
        # self.create_timer(1.0 / rate, self.control_loop)

        self.get_logger().info(
            f"velocity_adapter ready. agent={self.agent_name}, rate={rate} Hz"
        )

    # ------------------------------------------------------------------
    def world_cmd_cb(self, msg: TwistStamped):
        self.cmd_world[:] = [
            msg.twist.linear.x,
            msg.twist.linear.y,
            msg.twist.linear.z,
            msg.twist.angular.x,
            msg.twist.angular.y,
            msg.twist.angular.z,
        ]
        self.have_world_cmd = True

    def body_cmd_cb(self, msg: Twist):
        self.cmd_body[:] = [
            msg.linear.x,
            msg.linear.y,
            msg.linear.z,
            msg.angular.x,
            msg.angular.y,
            msg.angular.z,
        ]
        self.have_body_cmd = True

    def odom_cb(self, msg: Odometry):
        q = msg.pose.pose.orientation
        p = msg.pose.pose.position
        self.R_wb = quat_to_rot(q.x, q.y, q.z, q.w)
        self.p_world = np.array([p.x, p.y, p.z])
        self.have_odom = True

    def dvl_cb(self, msg: TwistWithCovarianceStamped):
        t = msg.twist.twist
        self.current_velocity[0:3] = [t.linear.x, t.linear.y, t.linear.z]
        self.got_dvl = True

    def imu_cb(self, msg: Imu):
        w = msg.angular_velocity
        self.current_velocity[3:6] = [w.x, w.y, w.z]
        self.got_imu = True

    # ------------------------------------------------------------------
    def _desired_body_velocity(self):
        """Convert command to body-frame velocity for the PID."""
        if self.have_world_cmd and self.have_odom:
            # Spatial twist (v_s, w_s) -> body twist (v_b, w_b)
            #   w_b = R^T w_s
            #   v_b = R^T (v_s + w_s x p)
            R_bw = self.R_wb.T
            v_s = self.cmd_world[0:3]
            # w_s = self.cmd_world[3:6]
            w_s = np.clip(self.cmd_world[3:6], -0.5, 0.5)
            p_w = self.p_world

            w_b = R_bw @ w_s
            v_b = R_bw @ (v_s + np.cross(w_s, p_w))
            return np.concatenate([v_b, w_b])
        if self.have_body_cmd:
            return self.cmd_body.copy()
        return np.zeros(6)

    def control_loop(self):
        now = self.get_clock().now()
        if self.last_time is None:
            self.last_time = now
            return
        dt = (now - self.last_time).nanoseconds * 1e-9
        self.last_time = now
        if dt <= 0.0 or not self.got_dvl:
            return

        # --- sim-time auto-stop ---
        t_sim = now.nanoseconds * 1e-9
        if self.t0 is None:
            self.t0 = t_sim
        elif self.max_sim_time > 0.0 and (t_sim - self.t0) >= self.max_sim_time:
            self.get_logger().info(
                f"[RUN] reached max_sim_time = {self.max_sim_time} s, stopping"
            )
            self.destroy_timer(self.timer)
            rclpy.shutdown()
            return
        v_des = self._desired_body_velocity()

        # Safety clamp on body-frame values (physical limits of the vehicle)
        v_des_preclamp = v_des.copy()
        # v_des[0:3] = np.clip(v_des[0:3], -self.max_v_lin, self.max_v_lin)
        # v_des[3:6] = np.clip(v_des[3:6], -self.max_v_ang, self.max_v_ang)

        error = v_des - self.current_velocity

        self.integral += error * dt
        # self.integral = np.clip(self.integral, -self.i_clamp, self.i_clamp)
        if not self.have_first_error:
            self.prev_error = error.copy()
            self.have_first_error = True
            derivative = np.zeros(6)
        else:
            raw_derivative = (error - self.prev_error) / dt
            alpha = 0.05  # higher = faster
            self.d_filtered = alpha * raw_derivative + (1 - alpha) * self.d_filtered
            derivative = self.d_filtered
            # derivative = (error - self.prev_error) / dt
            self.prev_error = error
        # derivative = (error - self.prev_error) / dt
        # self.prev_error = error

        tau = self.kp * error + self.ki * self.integral + self.kd * derivative
        forces_preclamp = np.array(T_PINV @ tau).reshape(-1)
        forces = np.clip(forces_preclamp, -MAX_THRUST, MAX_THRUST)
        if not np.all(np.isfinite(forces)):
            self.get_logger().warn(f"non-finite forces: {forces}")
            forces = np.zeros(8)
        # --- log one sample (sim time) ---
        self.log_t.append(t_sim)
        self.log_v_des.append(v_des.copy())
        self.log_v_cur.append(self.current_velocity.copy())
        self.log_err.append(error.copy())
        self.log_integral.append(self.integral.copy())
        self.log_derivative.append(derivative.copy())
        self.log_tau.append(tau.copy())
        self.log_forces.append(forces_preclamp.copy())

        cmd = AgentCommand()
        cmd.header.stamp = now.to_msg()
        cmd.header.frame_id = self.agent_name
        cmd.command = forces.tolist()
        self.pub.publish(cmd)

        if self.debug_log and (
            self.last_log_time is None or (now - self.last_log_time).nanoseconds > 1e9
        ):
            self.last_log_time = now
            self.get_logger().info(
                f"[AD] xi_s={np.round(self.cmd_world,3).tolist()} | "
                f"v_b_pre={np.round(v_des_preclamp,3).tolist()} | "
                f"v_des_body={np.round(v_des,3).tolist()} "
                f"v_cur={np.round(self.current_velocity,3).tolist()}"
                f"|v_err|={np.linalg.norm(v_des - self.current_velocity):.3f}"
            )

    def save_logs(self):
        out_dir = os.path.expanduser("~/holoocean_ws/logs")
        os.makedirs(out_dir, exist_ok=True)
        # path = os.path.join(out_dir, f"pid_trace_{self.run_stamp}.npz")
        path = os.path.join(out_dir, "pid_trace.npz")
        np.savez(
            path,
            t=np.array(self.log_t),
            v_des=np.array(self.log_v_des),
            v_cur=np.array(self.log_v_cur),
            err=np.array(self.log_err),
            integral=np.array(self.log_integral),
            derivative=np.array(self.log_derivative),
            tau=np.array(self.log_tau),
            forces=np.array(self.log_forces),
        )
        self.get_logger().info(f"[LOG] saved {len(self.log_t)} samples to {path}")


def main(args=None):
    rclpy.init(args=args)
    node = VelocityAdapter()
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
