#!/usr/bin/env python3

import rclpy

from geometry_msgs.msg import Twist
from rclpy.node import Node

from robot_interfaces.msg import TaskStatus


NAVIGATION_STATES = {
    "NAVIGATING_TO_OBJECT",
    "NAVIGATING_TO_ZONE",
}


class MockNavigationNode(Node):

    def __init__(self):
        super().__init__("mock_navigation_node")

        self.declare_parameter(
            "navigation_speed",
            0.20,
        )

        self.navigation_speed = (
            self.get_parameter("navigation_speed")
            .get_parameter_value()
            .double_value
        )

        self.status_subscription = (
            self.create_subscription(
                TaskStatus,
                "/task/status",
                self.status_callback,
                10,
            )
        )

        self.velocity_publisher = (
            self.create_publisher(
                Twist,
                "/cmd_vel_raw",
                10,
            )
        )

        self.timer = self.create_timer(
            0.1,
            self.publish_velocity,
        )

        self.current_state = "IDLE"
        self.last_reported_state = ""

        self.get_logger().info(
            "模拟导航速度节点已启动"
        )

    def status_callback(self, message):
        self.current_state = message.state

        if self.current_state != self.last_reported_state:
            self.get_logger().info(
                f"收到任务状态：{self.current_state}"
            )
            self.last_reported_state = (
                self.current_state
            )

    def publish_velocity(self):
        message = Twist()

        if self.current_state in NAVIGATION_STATES:
            message.linear.x = (
                self.navigation_speed
            )
            message.angular.z = 0.0
        else:
            message.linear.x = 0.0
            message.angular.z = 0.0

        self.velocity_publisher.publish(message)


def main(args=None):
    rclpy.init(args=args)

    node = MockNavigationNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        stop_message = Twist()
        node.velocity_publisher.publish(
            stop_message
        )

        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
