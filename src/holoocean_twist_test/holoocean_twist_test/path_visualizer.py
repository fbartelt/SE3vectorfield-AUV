#!/usr/bin/env python3
"""
Publishes the desired path and (optionally) the vehicle trail as
visualization_msgs/Marker messages to HoloOcean's debug/points topic.

The path is drawn from the SAME curve builder used by path_controller, so
the visualizer and the controller are guaranteed to agree.
"""

import numpy as np
import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from nav_msgs.msg import Odometry

# Adjust the import path if needed
from holoocean_twist_test.curves import build_curve


class PathVisualizer(Node):
    def __init__(self):
        super().__init__('path_visualizer')

        self.declare_parameter('agent_name', 'auv0')
        self.declare_parameter('curve_type', 'circle')
        self.declare_parameter('n_points', 200)     # visual only; keep low!
        # curve parameters (same names as path_controller)
        self.declare_parameter('radius', 1.5)
        self.declare_parameter('a', 2.0)
        self.declare_parameter('b', 1.0)
        self.declare_parameter('A', 2.0)
        self.declare_parameter('B', 1.5)
        self.declare_parameter('C', 0.5)
        self.declare_parameter('z0', -1.0)
        self.declare_parameter('center_x', 0.0)
        self.declare_parameter('center_y', 0.0)
        # visualization options
        self.declare_parameter('show_trail', True)
        self.declare_parameter('trail_max_len', 200)
        self.declare_parameter('republish_hz', 2.0)

        self.agent_name  = self.get_parameter('agent_name').value
        self.show_trail  = self.get_parameter('show_trail').value
        self.trail_max   = int(self.get_parameter('trail_max_len').value)
        republish_hz     = float(self.get_parameter('republish_hz').value)

        # --- Build the path using the SAME curve builder as the controller ---
        curve_type = self.get_parameter('curve_type').value
        kwargs = {
            "n_points": int(self.get_parameter('n_points').value),
            "z0":       float(self.get_parameter('z0').value),
            "center_xy": (float(self.get_parameter('center_x').value),
                          float(self.get_parameter('center_y').value)),
        }
        if curve_type == "circle":
            kwargs["radius"] = float(self.get_parameter('radius').value)
        elif curve_type == "ellipse":
            kwargs["a"] = float(self.get_parameter('a').value)
            kwargs["b"] = float(self.get_parameter('b').value)
        elif curve_type == "lissajous_3d":
            kwargs["A"] = float(self.get_parameter('A').value)
            kwargs["B"] = float(self.get_parameter('B').value)
            kwargs["C"] = float(self.get_parameter('C').value)

        curve, _ = build_curve(curve_type, **kwargs)   # (n, 4, 4)

        # --- Path marker: LINE_STRIP through the position part of each HTM ---
        path_marker = Marker()
        path_marker.header.frame_id = 'holoocean_global_frame'
        path_marker.ns = 'path'
        path_marker.id = 0
        path_marker.type = Marker.LINE_STRIP
        path_marker.action = Marker.ADD
        path_marker.scale.x = 0.05
        path_marker.color.r = 1.0
        path_marker.color.g = 0.1
        path_marker.color.b = 0.1
        path_marker.color.a = 1.0
        path_marker.pose.orientation.w = 1.0

        # Close the strip by appending the first point at the end
        pts = list(curve[:, :3, 3]) + [curve[0, :3, 3]]
        for (x, y, z) in pts:
            p = Point()
            p.x, p.y, p.z = float(x), float(y), float(z)
            path_marker.points.append(p)

        self.path_marker = path_marker

        # --- Trail marker ---
        self.trail_pts = []
        self.trail_marker = Marker()
        self.trail_marker.header.frame_id = 'holoocean_global_frame'
        self.trail_marker.ns = 'trail'
        self.trail_marker.id = 1
        self.trail_marker.type = Marker.LINE_STRIP
        self.trail_marker.action = Marker.ADD
        self.trail_marker.scale.x = 0.04
        self.trail_marker.color.r = 0.1
        self.trail_marker.color.g = 1.0
        self.trail_marker.color.b = 0.1
        self.trail_marker.color.a = 1.0
        self.trail_marker.pose.orientation.w = 1.0

        # --- ROS ---
        self.pub = self.create_publisher(Marker, 'debug/points', 10)

        if self.show_trail:
            self.create_subscription(
                Odometry,
                f'{self.agent_name}/DynamicsSensorOdom',
                self.odom_cb, 10)

        self.create_timer(1.0 / republish_hz, self.publish_markers)

        self.get_logger().info(
            f'path_visualizer ready. curve={curve_type}, '
            f'n_points={self.get_parameter("n_points").value}, '
            f'trail={self.show_trail}'
        )

    def odom_cb(self, msg: Odometry):
        p = msg.pose.pose.position
        self.trail_pts.append((p.x, p.y, p.z))
        if len(self.trail_pts) > self.trail_max:
            self.trail_pts.pop(0)

    def publish_markers(self):
        stamp = self.get_clock().now().to_msg()
        self.path_marker.header.stamp = stamp
        self.pub.publish(self.path_marker)

        if self.show_trail and len(self.trail_pts) >= 2:
            self.trail_marker.header.stamp = stamp
            self.trail_marker.points.clear()
            for (x, y, z) in self.trail_pts:
                pt = Point()
                pt.x, pt.y, pt.z = float(x), float(y), float(z)
                self.trail_marker.points.append(pt)
            self.pub.publish(self.trail_marker)


def main(args=None):
    rclpy.init(args=args)
    node = PathVisualizer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
