#!/usr/bin/env python3
"""
Twist -> Thruster adapter for HoloOcean (BlueROV2 / HoveringAUV).

Subscribes:
    /cmd_vel                                  geometry_msgs/Twist
    <ns>/auv0/DVLSensorVelocity               geometry_msgs/TwistWithCovarianceStamped  (linear)
    <ns>/auv0/IMUSensor                       sensor_msgs/Imu                          (angular)

Publishes:
    <ns>/command/agent                        holoocean_interfaces/AgentCommand
"""

import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, TwistWithCovarianceStamped
from sensor_msgs.msg import Imu
from holoocean_interfaces.msg import AgentCommand


# --- 6x8 thruster configuration matrix T, ROS FLU body frame ---
# Rows: [Fx, Fy, Fz, Mx, My, Mz]
# Cols: thrusters 0..7
T = np.array([
    [ 0.0,      0.0,      0.0,      0.0,      0.70710678,  0.70710678,  0.70710678,  0.70710678],
    [ 0.0,      0.0,      0.0,      0.0,     -0.70710678,  0.70710678, -0.70710678,  0.70710678],
    [ 1.0,      1.0,      1.0,      1.0,      0.0,         0.0,         0.0,         0.0       ],
    [-0.221,    0.221,    0.221,   -0.221,    0.0,         0.0,         0.0,         0.0       ],
    [-0.248,   -0.248,    0.248,    0.248,    0.0,         0.0,         0.0,         0.0       ],
    [ 0.0,      0.0,      0.0,      0.0,      0.0297,     -0.0297,     -0.0297,      0.0297    ],
])
T_PINV = np.linalg.pinv(T)

MAX_THRUST = 77.55  # N per thruster (from AUV_MAX_THRUST in HoveringAUV.h)


class TwistToAgent(Node):
    def __init__(self):
        super().__init__('twist_to_agent')

        self.declare_parameter('agent_name', 'auv0')
        self.declare_parameter('control_rate_hz', 50.0)
        self.declare_parameter('debug_log', True)

        self.agent_name = self.get_parameter('agent_name').value
        rate = self.get_parameter('control_rate_hz').value
        self.debug_log = self.get_parameter('debug_log').value

        # PID gains (6 DOF, body frame): [u, v, w, p, q, r]
        self.kp = np.array([60.0, 60.0, 60.0, 4.0, 4.0, 4.0])
        self.ki = np.array([ 0.0,  0.0,  0.0, 0.0, 0.0, 0.0])
        self.kd = np.array([ 5.0,  5.0,  5.0, 0.3, 0.3, 0.3])
        self.i_clamp = np.array([5.0, 5.0, 5.0, 1.0, 1.0, 1.0])

        self.integral = np.zeros(6)
        self.prev_error = np.zeros(6)

        self.desired_velocity = np.zeros(6)
        self.current_velocity = np.zeros(6)
        self.got_dvl = False
        self.got_imu = False

        self.last_time = None
        self.last_log_time = None

        # --- ROS interfaces ---
        self.create_subscription(Twist, '/cmd_vel', self.twist_cb, 10)

        # DVL -> linear velocity (body frame)
        self.create_subscription(
            TwistWithCovarianceStamped,
            f'{self.agent_name}/DVLSensorVelocity',
            self.dvl_cb,
            10,
        )

        # IMU -> angular velocity (body frame)
        self.create_subscription(
            Imu,
            f'{self.agent_name}/IMUSensor',
            self.imu_cb,
            10,
        )

        self.pub = self.create_publisher(AgentCommand, 'command/agent', 10)
        self.create_timer(1.0 / rate, self.control_loop)

        self.get_logger().info(
            f'twist_to_agent ready. agent={self.agent_name}, rate={rate} Hz'
        )

    # ------------------------------------------------------------------
    def twist_cb(self, msg: Twist):
        self.desired_velocity[:] = [
            msg.linear.x, msg.linear.y, msg.linear.z,
            msg.angular.x, msg.angular.y, msg.angular.z,
        ]

    def dvl_cb(self, msg: TwistWithCovarianceStamped):
        t = msg.twist.twist
        self.current_velocity[0] = t.linear.x
        self.current_velocity[1] = t.linear.y
        self.current_velocity[2] = t.linear.z
        self.got_dvl = True

    def imu_cb(self, msg: Imu):
        w = msg.angular_velocity
        self.current_velocity[3] = w.x
        self.current_velocity[4] = w.y
        self.current_velocity[5] = w.z
        self.got_imu = True

    # ------------------------------------------------------------------
    def control_loop(self):
        now = self.get_clock().now()
        if self.last_time is None:
            self.last_time = now
            return
        dt = (now - self.last_time).nanoseconds * 1e-9
        self.last_time = now
        if dt <= 0.0:
            return

        # Require at least the DVL to have reported; IMU comes in quickly after.
        if not self.got_dvl:
            return

        error = self.desired_velocity - self.current_velocity

        # PI-D (ki currently zero -> pure PD)
        self.integral += error * dt
        self.integral = np.clip(self.integral, -self.i_clamp, self.i_clamp)
        derivative = (error - self.prev_error) / dt
        self.prev_error = error

        tau = self.kp * error + self.ki * self.integral + self.kd * derivative

        # Control allocation
        forces = T_PINV @ tau
        forces = np.clip(forces, -MAX_THRUST, MAX_THRUST)

        cmd = AgentCommand()
        cmd.header.stamp = now.to_msg()
        cmd.header.frame_id = self.agent_name
        cmd.command = forces.tolist()
        self.pub.publish(cmd)

        if self.debug_log:
            # Log once every second
            if self.last_log_time is None or (now - self.last_log_time).nanoseconds > 1e9:
                self.last_log_time = now
                self.get_logger().info(
                    f'v_des={np.round(self.desired_velocity, 3).tolist()} '
                    f'v_cur={np.round(self.current_velocity, 3).tolist()} '
                    f'f={np.round(forces, 2).tolist()}'
                )


def main(args=None):
    rclpy.init(args=args)
    node = TwistToAgent()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()
