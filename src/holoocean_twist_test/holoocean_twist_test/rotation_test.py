#!/usr/bin/env python3
"""
Rotation test: regulation or constant-velocity tracking on a single
body axis (roll, pitch, or yaw). No curve involved.

Publishes:
    /cmd_vel    geometry_msgs/Twist  (body-frame, adapter will use it)

Subscribes:
    /holoocean/auv0/DynamicsSensorOdom  nav_msgs/Odometry
"""

import os
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry


def quat_to_euler_zyx(x, y, z, w):
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    pitch = np.arcsin(np.clip(sinp, -1.0, 1.0))

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.array([roll, pitch, yaw])


class RotationTest(Node):
    def __init__(self):
        super().__init__('rotation_test')

        self.declare_parameter('agent_name', 'auv0')
        self.declare_parameter('axis', 'yaw')          # 'roll' | 'pitch' | 'yaw'
        self.declare_parameter('mode', 'regulate')     # 'regulate' | 'velocity'
        self.declare_parameter('target_angle', 0.5)    # rad (regulate)
        self.declare_parameter('target_omega', 0.5)    # rad/s (velocity)
        self.declare_parameter('kp', 1.0)
        self.declare_parameter('rate_hz', 20.0)
        self.declare_parameter('max_sim_time', 20.0)

        self.agent_name  = self.get_parameter('agent_name').value
        self.axis        = self.get_parameter('axis').value
        self.mode        = self.get_parameter('mode').value
        self.target_angle = float(self.get_parameter('target_angle').value)
        self.target_omega = float(self.get_parameter('target_omega').value)
        self.kp          = float(self.get_parameter('kp').value)
        self.max_sim_time = float(self.get_parameter('max_sim_time').value)
        rate             = float(self.get_parameter('rate_hz').value)

        self.axis_idx = {'roll': 0, 'pitch': 1, 'yaw': 2}[self.axis]

        self.euler = np.zeros(3)
        self.omega_cur = np.zeros(3)
        self.have_pose = False
        self.t0 = None
        self.last_time = None

        # logging
        self.log_t = []
        self.log_euler = []
        self.log_omega = []
        self.log_omega_des = []

        self.create_subscription(
            Odometry,
            f'/holoocean/{self.agent_name}/DynamicsSensorOdom',
            self.odom_cb, 10,
        )
        self.pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.timer = self.create_timer(1.0 / rate, self.loop)

        self.get_logger().info(
            f'rotation_test ready: axis={self.axis} mode={self.mode} '
            f'target_angle={self.target_angle:.2f} '
            f'target_omega={self.target_omega:.2f} kp={self.kp}'
        )

    def odom_cb(self, msg):
        q = msg.pose.pose.orientation
        self.euler = quat_to_euler_zyx(q.x, q.y, q.z, q.w)
        w = msg.twist.twist.angular
        self.omega_cur = np.array([w.x, w.y, w.z])
        self.have_pose = True

    def loop(self):
        if not self.have_pose:
            return

        now = self.get_clock().now()
        t_sim = now.nanoseconds * 1e-9

        if self.t0 is None:
            self.t0 = t_sim
            self.last_time = now
            return
        if self.max_sim_time > 0.0 and (t_sim - self.t0) >= self.max_sim_time:
            self.get_logger().info('[RUN] reached max_sim_time')
            self.destroy_timer(self.timer)
            rclpy.shutdown()
            return

        # Desired body-frame angular velocity on the selected axis
        if self.mode == 'regulate':
            err = self.target_angle - self.euler[self.axis_idx]
            # wrap to [-pi, pi]
            err = (err + np.pi) % (2.0 * np.pi) - np.pi
            omega_des = self.kp * err
        else:  # velocity
            omega_des = self.target_omega

        msg = Twist()
        # linear stays zero: pure rotation, no translation
        msg.linear.x = 0.0
        msg.linear.y = 0.0
        msg.linear.z = 0.0
        msg.angular.x = 0.0
        msg.angular.y = 0.0
        msg.angular.z = 0.0

        if self.axis_idx == 0:
            msg.angular.x = float(omega_des)
        elif self.axis_idx == 1:
            msg.angular.y = float(omega_des)
        else:
            msg.angular.z = float(omega_des)

        self.pub.publish(msg)

        # logging
        self.log_t.append(t_sim)
        self.log_euler.append(self.euler.copy())
        self.log_omega.append(self.omega_cur.copy())
        self.log_omega_des.append(
            np.array([msg.angular.x, msg.angular.y, msg.angular.z])
        )

    def save_logs(self):
        out_dir = os.path.expanduser('~/holoocean_ws/logs')
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(
            out_dir, f'rotation_test_{self.axis}_{self.mode}.npz'
        )
        np.savez(
            path,
            t=np.array(self.log_t),
            euler=np.array(self.log_euler),
            omega=np.array(self.log_omega),
            omega_des=np.array(self.log_omega_des),
        )
        self.get_logger().info(f'[LOG] saved {len(self.log_t)} samples to {path}')


def main(args=None):
    rclpy.init(args=args)
    node = RotationTest()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.save_logs()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
