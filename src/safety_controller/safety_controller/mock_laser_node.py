#!/usr/bin/env python3

import math

import rclpy

from rclpy.node import Node
from sensor_msgs.msg import LaserScan


class MockLaserNode(Node):

    def __init__(self):
        super().__init__("mock_laser_node")

        self.declare_parameter(
            "obstacle_distance",
            3.0,
        )

        self.declare_parameter(
            "scan_rate_hz",
            10.0,
        )

        scan_rate = (
            self.get_parameter("scan_rate_hz")
            .get_parameter_value()
            .double_value
        )

        self.publisher = self.create_publisher(
            LaserScan,
            "/scan",
            10,
        )

        self.timer_period = 1.0 / scan_rate

        self.timer = self.create_timer(
            self.timer_period,
            self.publish_scan,
        )

        self.last_distance = None

        self.get_logger().info(
            "模拟激光雷达已启动"
        )

    def publish_scan(self):
        range_min = 0.10
        range_max = 10.0
        sample_count = 360

        obstacle_distance = (
            self.get_parameter("obstacle_distance")
            .get_parameter_value()
            .double_value
        )

        obstacle_distance = max(
            range_min,
            min(range_max, obstacle_distance),
        )

        if obstacle_distance != self.last_distance:
            self.get_logger().info(
                f"前方模拟障碍距离："
                f"{obstacle_distance:.2f}米"
            )
            self.last_distance = obstacle_distance

        angle_min = -math.pi
        angle_max = math.pi
        angle_increment = (
            angle_max - angle_min
        ) / sample_count

        ranges = [range_max] * sample_count

        front_half_angle = math.radians(15.0)

        for index in range(sample_count):
            angle = (
                angle_min
                + index * angle_increment
            )

            if abs(angle) <= front_half_angle:
                ranges[index] = obstacle_distance

        message = LaserScan()

        message.header.stamp = (
            self.get_clock().now().to_msg()
        )
        message.header.frame_id = "laser_frame"

        message.angle_min = angle_min
        message.angle_max = angle_max
        message.angle_increment = angle_increment

        message.time_increment = 0.0
        message.scan_time = self.timer_period

        message.range_min = range_min
        message.range_max = range_max

        message.ranges = ranges
        message.intensities = []

        self.publisher.publish(message)


def main(args=None):
    rclpy.init(args=args)

    node = MockLaserNode()

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
