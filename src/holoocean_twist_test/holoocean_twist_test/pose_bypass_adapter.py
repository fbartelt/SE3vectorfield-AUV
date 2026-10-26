#!/usr/bin/env python3
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped, Twist
from nav_msgs.msg import Odometry
from holoocean_interfaces.msg import AgentCommand
from scipy.spatial.transform import Rotation as R


class PoseBypassAdapter(Node):
    def __init__(self):
        super().__init__("pose_bypass_adapter")

        self.declare_parameter("agent_name", "auv0")
        self.agent_name = self.get_parameter("agent_name").value

        self.pos = None
        self.rot = None
        self.have_odom = False

        self.last_time = None
        self.latest_twist = Twist()

        # 1. Listen to world commands (from path_controller)
        self.create_subscription(
            TwistStamped, "/controller/cmd_vel_world", self.world_cmd_cb, 10
        )

        # 2. Listen to body commands (from rotation_test)
        self.create_subscription(Twist, "/cmd_vel", self.body_cmd_cb, 10)

        # 3. Listen to odometry to get the spawn pose
        # Note: using relative topic name to resolve correctly under the namespace
        self.create_subscription(
            Odometry, f"{self.agent_name}/DynamicsSensorOdom", self.odom_cb, 10
        )

        # 4. Publish directly to HoloOcean's Control Scheme 1
        self.pub = self.create_publisher(AgentCommand, "command/agent", 10)

        self.create_timer(0.05, self.loop)  # 20 Hz

        self.get_logger().info("Kinematic Bypass Adapter ready (Scheme 1).")

    def odom_cb(self, msg):
        if not self.have_odom:
            p = msg.pose.pose.position
            q = msg.pose.pose.orientation
            self.pos = np.array([p.x, p.y, p.z])
            self.rot = R.from_quat([q.x, q.y, q.z, q.w])
            self.have_odom = True
            self.get_logger().info(
                f"Initialized pose target to: {np.round(self.pos, 2)}"
            )

    def world_cmd_cb(self, msg):
        self.latest_twist = msg.twist
        # Note: if you ever mix this with the body twist, you would need to
        # transform it by the current rotation. For now, we assume one or the other.

    def body_cmd_cb(self, msg):
        self.latest_twist = msg

    def loop(self):
        # We only need odometry to start publishing (to hold position)
        if not self.have_odom:
            return

        now = self.get_clock().now()
        if self.last_time is None:
            self.last_time = now
            return

        dt = (now - self.last_time).nanoseconds * 1e-9
        self.last_time = now

        if dt <= 0.0:
            return

        # Integrate linear velocity
        v = np.array(
            [
                self.latest_twist.linear.x,
                self.latest_twist.linear.y,
                self.latest_twist.linear.z,
            ]
        )
        self.pos += v * dt

        # Integrate angular velocity
        w = np.array(
            [
                self.latest_twist.angular.x,
                self.latest_twist.angular.y,
                self.latest_twist.angular.z,
            ]
        )
        if np.linalg.norm(w) > 1e-6:
            delta_rot = R.from_rotvec(w * dt)
            self.rot = delta_rot * self.rot

        # Convert to Euler angles and then to DEGREES
        # HoloOcean's Scheme 1 uses Unreal's FRotator, which expects degrees.
        euler_rad = self.rot.as_euler("xyz")  # roll, pitch, yaw
        euler_deg = np.degrees(euler_rad)

        cmd = AgentCommand()
        cmd.header.stamp = now.to_msg()
        cmd.header.frame_id = self.agent_name

        # [x, y, z, roll, pitch, yaw] in degrees
        cmd.command = [
            float(self.pos[0]),
            float(self.pos[1]),
            float(self.pos[2]),
            float(euler_deg[0]),
            float(euler_deg[1]),
            float(euler_deg[2]),
        ]
        self.pub.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = PoseBypassAdapter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
