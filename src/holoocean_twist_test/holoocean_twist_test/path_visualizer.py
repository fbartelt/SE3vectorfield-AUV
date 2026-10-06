#!/usr/bin/env python3
"""
Publishes the desired path and (optionally) the vehicle trail as
visualization_msgs/Marker messages to HoloOcean's debug/points topic.
"""

import numpy as np
import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from nav_msgs.msg import Odometry


def circle_path(radius, z0, center_xy, n_points, n_rolls):
    """Same parametrization as path_controller.circle_with_barrel_roll."""
    cx, cy = center_xy
    pts = np.zeros((n_points, 3))
    for i in range(n_points):
        s = 2.0 * np.pi * i / n_points
        pts[i] = (cx + radius * np.cos(s),
                  cy + radius * np.sin(s),
                  z0)
    return pts


class PathVisualizer(Node):
    def __init__(self):
        super().__init__('path_visualizer')

        self.declare_parameter('agent_name', 'auv0')
        self.declare_parameter('radius', 1.5)
        self.declare_parameter('z0', -3.0)
        self.declare_parameter('center_x', 0.0)
        self.declare_parameter('center_y', 0.0)
        self.declare_parameter('n_points', 300)
        self.declare_parameter('n_rolls', 0.0)
        self.declare_parameter('show_trail', True)
        self.declare_parameter('trail_max_len', 500)
        self.declare_parameter('republish_hz', 1.0)

        self.agent_name   = self.get_parameter('agent_name').value
        self.show_trail   = self.get_parameter('show_trail').value
        self.trail_max    = int(self.get_parameter('trail_max_len').value)
        republish_hz      = float(self.get_parameter('republish_hz').value)

        # --- build path marker ---
        pts = circle_path(
            radius=float(self.get_parameter('radius').value),
            z0=float(self.get_parameter('z0').value),
            center_xy=(float(self.get_parameter('center_x').value),
                       float(self.get_parameter('center_y').value)),
            n_points=int(self.get_parameter('n_points').value),
            n_rolls=float(self.get_parameter('n_rolls').value),
        )

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
        for (x, y, z) in pts:
            p = Point()
            p.x, p.y, p.z = float(x), float(y), float(z)
            path_marker.points.append(p)

        self.path_marker = path_marker

        # --- trail marker (built incrementally) ---
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

        # Republish at fixed rate: HoloOcean re-renders each tick, and this
        # keeps the markers alive if the viewport resets.
        self.create_timer(1.0 / republish_hz, self.publish_markers)

        self.get_logger().info(
            f'path_visualizer ready. radius={self.get_parameter("radius").value} '
            f'z0={self.get_parameter("z0").value} '
            f'center=({self.get_parameter("center_x").value},{self.get_parameter("center_y").value}) '
            f'trail={self.show_trail}'
        )

    def odom_cb(self, msg: Odometry):
        p = msg.pose.pose.position
        self.trail_pts.append((p.x, p.y, p.z))
        if len(self.trail_pts) > self.trail_max:
            self.trail_pts.pop(0)

    def publish_markers(self):
        self.path_marker.header.stamp = self.get_clock().now().to_msg()
        self.pub.publish(self.path_marker)

        if self.show_trail and len(self.trail_pts) >= 2:
            self.trail_marker.header.stamp = self.get_clock().now().to_msg()
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
