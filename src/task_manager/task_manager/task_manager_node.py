#!/usr/bin/env python3

import rclpy

from rclpy.node import Node

from robot_interfaces.msg import ObjectDetectionArray
from robot_interfaces.msg import TaskCommand
from robot_interfaces.msg import TaskStatus


ALLOWED_COLORS = {
    "red",
    "green",
    "blue",
    "yellow",
}

ALLOWED_DESTINATIONS = {
    "A",
    "B",
    "C",
}


class TaskManagerNode(Node):

    def __init__(self):
        super().__init__("task_manager_node")

        self.command_subscription = self.create_subscription(
            TaskCommand,
            "/llm/task_command",
            self.command_callback,
            10,
        )

        self.vision_subscription = self.create_subscription(
            ObjectDetectionArray,
            "/vision/detections",
            self.vision_callback,
            10,
        )

        self.status_publisher = self.create_publisher(
            TaskStatus,
            "/task/status",
            10,
        )

        self.timer = self.create_timer(
            1.0,
            self.timer_callback,
        )

        self.state = "IDLE"
        self.active = False

        self.target_color = ""
        self.target_class = ""
        self.destination = ""

        self.current_iteration = 0
        self.total_iterations = 0

        self.visible_classes = set()
        self.locate_attempts = 0
        self.maximum_locate_attempts = 5

        self.publish_status(
            "IDLE",
            "任务管理器已启动，等待任务",
        )

        self.get_logger().info(
            "模拟任务状态机已启动"
        )

    def publish_status(
        self,
        state,
        detail,
        success=False,
    ):
        self.state = state

        message = TaskStatus()
        message.header.stamp = (
            self.get_clock().now().to_msg()
        )
        message.header.frame_id = "task_manager"

        message.state = state
        message.detail = detail

        message.current_iteration = (
            self.current_iteration
        )
        message.total_iterations = (
            self.total_iterations
        )

        message.success = success

        self.status_publisher.publish(message)

        self.get_logger().info(
            f"[{state}] {detail}"
        )

    def vision_callback(self, message):
        self.visible_classes = {
            detection.class_name
            for detection in message.detections
        }

    def validate_task(self, message):
        if message.action != "pick_and_place":
            return False, "不支持的任务动作"

        if message.color not in ALLOWED_COLORS:
            return False, "不支持的物块颜色"

        if message.destination not in ALLOWED_DESTINATIONS:
            return False, "不支持的目标区域"

        if not 1 <= message.count <= 5:
            return False, "执行次数必须在1到5之间"

        return True, "任务合法"

    def command_callback(self, message):
        if message.action == "stop":
            self.active = False

            self.publish_status(
                "EMERGENCY_STOP",
                "收到停止指令，模拟任务已停止",
            )
            return

        if self.active:
            self.get_logger().warning(
                "当前已有任务正在执行，拒绝新任务"
            )
            return

        valid, reason = self.validate_task(message)

        if not valid:
            self.publish_status(
                "FAILED",
                reason,
            )
            return

        self.target_color = message.color
        self.target_class = (
            f"{message.color}_block"
        )
        self.destination = message.destination

        self.current_iteration = 0
        self.total_iterations = message.count

        self.locate_attempts = 0
        self.active = True

        self.publish_status(
            "LOCATING_OBJECT",
            f"开始寻找{self.target_class}",
        )

    def timer_callback(self):
        if not self.active:
            return

        if self.state == "LOCATING_OBJECT":
            if self.target_class in self.visible_classes:
                self.locate_attempts = 0

                self.publish_status(
                    "NAVIGATING_TO_OBJECT",
                    f"已找到{self.target_class}，"
                    "模拟导航到物块",
                )
            else:
                self.locate_attempts += 1

                if (
                    self.locate_attempts
                    >= self.maximum_locate_attempts
                ):
                    self.active = False

                    self.publish_status(
                        "FAILED",
                        f"未找到{self.target_class}",
                    )
                else:
                    self.publish_status(
                        "LOCATING_OBJECT",
                        f"正在寻找{self.target_class}，"
                        f"第{self.locate_attempts}次",
                    )

        elif self.state == "NAVIGATING_TO_OBJECT":
            self.publish_status(
                "GRASPING",
                "模拟导航完成，正在抓取",
            )

        elif self.state == "GRASPING":
            self.publish_status(
                "NAVIGATING_TO_ZONE",
                f"模拟抓取成功，前往{self.destination}区",
            )

        elif self.state == "NAVIGATING_TO_ZONE":
            self.publish_status(
                "PLACING",
                f"已到达{self.destination}区，正在放置",
            )

        elif self.state == "PLACING":
            self.current_iteration += 1

            if (
                self.current_iteration
                < self.total_iterations
            ):
                self.locate_attempts = 0

                self.publish_status(
                    "LOCATING_OBJECT",
                    f"第{self.current_iteration}次完成，"
                    "开始下一次",
                )
            else:
                self.active = False

                self.publish_status(
                    "FINISHED",
                    f"全部{self.total_iterations}次任务完成",
                    success=True,
                )


def main(args=None):
    rclpy.init(args=args)

    node = TaskManagerNode()

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
