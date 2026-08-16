#!/usr/bin/env python3

import math

import rclpy

from geometry_msgs.msg import Twist
from rclpy.node import Node
from sensor_msgs.msg import LaserScan

from robot_interfaces.msg import TaskStatus
from robot_interfaces.srv import EmergencyStop


class SafetyFilterNode(Node):

    def __init__(self):
        super().__init__("safety_filter_node")

        self.declare_parameter("stop_distance", 0.60)
        self.declare_parameter("slow_distance", 1.20)
        self.declare_parameter("scan_timeout", 0.50)
        self.declare_parameter("max_linear_speed", 0.30)
        self.declare_parameter("max_angular_speed", 0.80)

        self.stop_distance = self.get_parameter(
            "stop_distance"
        ).value

        self.slow_distance = self.get_parameter(
            "slow_distance"
        ).value

        self.scan_timeout = self.get_parameter(
            "scan_timeout"
        ).value

        self.max_linear_speed = self.get_parameter(
            "max_linear_speed"
        ).value

        self.max_angular_speed = self.get_parameter(
            "max_angular_speed"
        ).value

        self.scan_subscription = self.create_subscription(
            LaserScan,
            "/scan",
            self.scan_callback,
            10,
        )
        self.task_status_subscription = (
            self.create_subscription(
                TaskStatus,
                "/task/status",
                self.task_status_callback,
                10,
            )
        )

        self.raw_velocity_subscription = (
            self.create_subscription(
                Twist,
                "/cmd_vel_raw",
                self.velocity_callback,
                10,
            )
        )

        self.safe_velocity_publisher = (
            self.create_publisher(
                Twist,
                "/cmd_vel",
                10,
            )
        )

        self.emergency_stop_service = (
            self.create_service(
                EmergencyStop,
                "/safety/emergency_stop",
                self.emergency_stop_callback,
            )
        )

        self.timer = self.create_timer(
            0.1,
            self.publish_safe_velocity,
        )

        self.latest_raw_velocity = Twist()
        self.front_distance = math.inf
        self.last_scan_time = None

        self.emergency_stop_active = False
        self.last_safety_state = ""

        self.get_logger().info(
            "速度安全过滤器已启动"
        )

    def scan_callback(self, message):
        front_values = []

        for index, distance in enumerate(
            message.ranges
        ):
            angle = (
                message.angle_min
                + index * message.angle_increment
            )

            if abs(angle) > math.radians(30.0):
                continue

            if not math.isfinite(distance):
                continue

            if (
                message.range_min
                <= distance
                <= message.range_max
            ):
                front_values.append(distance)

        if front_values:
            self.front_distance = min(front_values)
        else:
            self.front_distance = math.inf

        self.last_scan_time = self.get_clock().now()

    def velocity_callback(self, message):
        self.latest_raw_velocity = message

    def scan_is_fresh(self):
        if self.last_scan_time is None:
            return False

        elapsed = (
            self.get_clock().now()
            - self.last_scan_time
        ).nanoseconds / 1_000_000_000.0

        return elapsed <= self.scan_timeout

    def limit_value(self, value, maximum):
        return max(
            -maximum,
            min(maximum, value),
        )

    def create_zero_velocity(self):
        return Twist()

    def publish_safety_state(self, state, detail):
        if state == self.last_safety_state:
            return

        self.last_safety_state = state

        if state in {
            "BLOCKED",
            "NO_SCAN",
            "EMERGENCY_STOP",
        }:
            self.get_logger().warning(
                f"[{state}] {detail}"
            )
        else:
            self.get_logger().info(
                f"[{state}] {detail}"
            )

    def publish_safe_velocity(self):
        if self.emergency_stop_active:
            safe_velocity = (
                self.create_zero_velocity()
            )

            self.publish_safety_state(
                "EMERGENCY_STOP",
                "急停已触发，速度强制为零",
            )

        elif not self.scan_is_fresh():
            safe_velocity = (
                self.create_zero_velocity()
            )

            self.publish_safety_state(
                "NO_SCAN",
                "雷达数据超时，速度强制为零",
            )

        elif (
            self.front_distance
            <= self.stop_distance
        ):
            safe_velocity = (
                self.create_zero_velocity()
            )

            self.publish_safety_state(
                "BLOCKED",
                f"前方障碍距离"
                f"{self.front_distance:.2f}米，停车",
            )

        else:
            safe_velocity = Twist()

            raw_linear = self.limit_value(
                self.latest_raw_velocity.linear.x,
                self.max_linear_speed,
            )

            raw_angular = self.limit_value(
                self.latest_raw_velocity.angular.z,
                self.max_angular_speed,
            )

            if (
                raw_linear > 0.0
                and self.front_distance
                < self.slow_distance
            ):
                scale = (
                    self.front_distance
                    - self.stop_distance
                ) / (
                    self.slow_distance
                    - self.stop_distance
                )

                scale = max(
                    0.0,
                    min(1.0, scale),
                )

                safe_velocity.linear.x = (
                    raw_linear * scale
                )
                safe_velocity.angular.z = (
                    raw_angular
                )

                self.publish_safety_state(
                    "SLOW",
                    f"前方距离"
                    f"{self.front_distance:.2f}米，减速",
                )

            else:
                safe_velocity.linear.x = raw_linear
                safe_velocity.angular.z = raw_angular

                self.publish_safety_state(
                    "CLEAR",
                    f"前方距离"
                    f"{self.front_distance:.2f}米，允许通行",
                )

        self.safe_velocity_publisher.publish(
            safe_velocity
        )

    def task_status_callback(self, message):
        if message.state != "EMERGENCY_STOP":
            return

        if self.emergency_stop_active:
            return

        self.emergency_stop_active = True
        self.last_safety_state = ""

        self.get_logger().warning(
            "任务状态进入EMERGENCY_STOP，"
            "安全急停已自动锁定"
        )
    def emergency_stop_callback(
        self,
        request,
        response,
    ):
        if request.engage:
            self.emergency_stop_active = True

            response.accepted = True
            response.message = (
                f"急停已触发：{request.reason}"
            )

            return response

        if not self.scan_is_fresh():
            response.accepted = False
            response.message = (
                "雷达数据无效，拒绝解除急停"
            )

            return response

        if self.front_distance <= self.stop_distance:
            response.accepted = False
            response.message = (
                "前方障碍过近，拒绝解除急停"
            )

            return response

        self.emergency_stop_active = False
        self.last_safety_state = ""

        response.accepted = True
        response.message = "急停已人工解除"

        return response


def main(args=None):
    rclpy.init(args=args)

    node = SafetyFilterNode()

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
