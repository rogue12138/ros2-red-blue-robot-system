#!/usr/bin/env python3

import uuid

import rclpy

from rclpy.node import Node
from std_msgs.msg import String

from robot_interfaces.msg import ObjectDetectionArray
from robot_interfaces.msg import TaskCommand
from robot_interfaces.msg import TaskPlan
from robot_interfaces.msg import TaskProgress
from robot_interfaces.msg import TaskStatus
from robot_interfaces.msg import TaskStep


ALLOWED_COLORS = {
    "red",
    "blue",
}

ALLOWED_DESTINATIONS = {
    "A",
    "B",
    "C",
}


class TaskPlanManagerNode(Node):

    def __init__(self):
        super().__init__(
            "task_plan_manager_node"
        )

        self.declare_parameter(
            "maximum_locate_attempts",
            5,
        )
        self.declare_parameter(
            "max_step_count",
            8,
        )
        self.declare_parameter(
            "max_total_count",
            10,
        )
        self.declare_parameter(
            "enable_legacy_command",
            False,
        )

        self.maximum_locate_attempts = int(
            self.get_parameter(
                "maximum_locate_attempts"
            ).value
        )
        self.max_step_count = int(
            self.get_parameter(
                "max_step_count"
            ).value
        )
        self.max_total_count = int(
            self.get_parameter(
                "max_total_count"
            ).value
        )
        self.enable_legacy_command = bool(
            self.get_parameter(
                "enable_legacy_command"
            ).value
        )

        self.plan_subscription = (
            self.create_subscription(
                TaskPlan,
                "/llm/task_plan",
                self.plan_callback,
                10,
            )
        )

        self.vision_subscription = (
            self.create_subscription(
                ObjectDetectionArray,
                "/vision/detections",
                self.vision_callback,
                10,
            )
        )

        self.status_publisher = (
            self.create_publisher(
                TaskStatus,
                "/task/status",
                10,
            )
        )

        self.progress_publisher = (
            self.create_publisher(
                TaskProgress,
                "/task/progress",
                10,
            )
        )

        self.feedback_publisher = (
            self.create_publisher(
                String,
                "/task/feedback",
                10,
            )
        )

        self.legacy_subscription = None

        if self.enable_legacy_command:
            self.legacy_subscription = (
                self.create_subscription(
                    TaskCommand,
                    "/llm/task_command",
                    self.legacy_command_callback,
                    10,
                )
            )

        self.timer = self.create_timer(
            1.0,
            self.timer_callback,
        )

        self.state = "IDLE"
        self.active = False

        self.task_id = ""
        self.original_text = ""
        self.steps = []
        self.current_step_index = 0

        self.current_color = ""
        self.current_destination = ""
        self.target_classes = set()

        self.step_completed = 0
        self.step_total = 0
        self.total_completed = 0
        self.total_target = 0

        self.visible_classes = set()
        self.locate_attempts = 0

        self.publish_status(
            "IDLE",
            "多步骤任务管理器已启动，等待任务",
        )

        self.get_logger().info(
            "多步骤模拟任务状态机已启动"
        )

        if self.enable_legacy_command:
            self.get_logger().warning(
                "已启用旧TaskCommand兼容模式"
            )

    def publish_feedback(self, text):
        message = String()
        message.data = text
        self.feedback_publisher.publish(message)

        self.get_logger().warning(text)

    def publish_status(
        self,
        state,
        detail,
        success=False,
    ):
        self.state = state

        status_message = TaskStatus()

        status_message.header.stamp = (
            self.get_clock().now().to_msg()
        )
        status_message.header.frame_id = (
            "task_plan_manager"
        )

        status_message.state = state
        status_message.detail = detail

        status_message.current_iteration = (
            self.total_completed
        )
        status_message.total_iterations = (
            self.total_target
        )

        status_message.success = success

        self.status_publisher.publish(
            status_message
        )

        progress_message = TaskProgress()

        progress_message.header.stamp = (
            self.get_clock().now().to_msg()
        )
        progress_message.header.frame_id = (
            "task_plan_manager"
        )

        progress_message.task_id = self.task_id
        progress_message.state = state
        progress_message.detail = detail

        if self.steps:
            progress_message.current_step = min(
                self.current_step_index + 1,
                len(self.steps),
            )
        else:
            progress_message.current_step = 0

        progress_message.total_steps = len(
            self.steps
        )
        progress_message.current_color = (
            self.current_color
        )
        progress_message.current_destination = (
            self.current_destination
        )

        progress_message.step_completed = (
            self.step_completed
        )
        progress_message.step_total = (
            self.step_total
        )
        progress_message.total_completed = (
            self.total_completed
        )
        progress_message.total_target = (
            self.total_target
        )
        progress_message.success = success

        self.progress_publisher.publish(
            progress_message
        )

        self.get_logger().info(
            f"[{state}] {detail}"
        )

    def vision_callback(self, message):
        self.visible_classes = {
            detection.class_name
            for detection in message.detections
        }

    def validate_plan_steps(self, message):
        if not message.steps:
            return (
                False,
                "任务步骤列表为空",
                [],
                0,
            )

        if len(message.steps) > self.max_step_count:
            return (
                False,
                "任务步骤数量超过限制",
                [],
                0,
            )

        normalized_steps = []
        calculated_total = 0

        for index, step in enumerate(
            message.steps,
            start=1,
        ):
            color = step.color.strip().lower()
            destination = (
                step.destination.strip().upper()
            )
            count = int(step.count)

            if color not in ALLOWED_COLORS:
                return (
                    False,
                    f"第{index}步颜色不合法："
                    f"{color}",
                    [],
                    0,
                )

            if (
                destination
                not in ALLOWED_DESTINATIONS
            ):
                return (
                    False,
                    f"第{index}步区域不合法："
                    f"{destination}",
                    [],
                    0,
                )

            if not 1 <= count <= 8:
                return (
                    False,
                    f"第{index}步数量必须"
                    "在1到8之间",
                    [],
                    0,
                )

            normalized_steps.append(
                {
                    "color": color,
                    "destination": destination,
                    "count": count,
                }
            )

            calculated_total += count

        if (
            calculated_total
            > self.max_total_count
        ):
            return (
                False,
                "任务总数量超过限制："
                f"{calculated_total} > "
                f"{self.max_total_count}",
                [],
                0,
            )

        if (
            message.total_count
            not in {0, calculated_total}
        ):
            return (
                False,
                "任务总数与步骤数量之和不一致",
                [],
                0,
            )

        return (
            True,
            "任务合法",
            normalized_steps,
            calculated_total,
        )

    def handle_stop(self, reason):
        self.active = False

        if not reason:
            reason = "收到停止指令"

        self.publish_status(
            "EMERGENCY_STOP",
            reason,
        )

        self.publish_feedback(
            "任务已紧急停止，"
            "人工解除安全锁定后也不会自动恢复"
        )

    def plan_callback(self, message):
        action = (
            message.action.strip().lower()
        )

        if action == "stop":
            self.handle_stop(
                message.reason
            )
            return

        if action == "clarify":
            self.publish_feedback(
                "任务信息需要补充："
                f"{message.reason}"
            )
            return

        if action == "reject":
            self.publish_feedback(
                "任务已被语言节点拒绝："
                f"{message.reason}"
            )
            return

        if action != "execute":
            self.publish_feedback(
                f"不支持的任务动作：{action}"
            )
            return

        if self.active:
            self.publish_feedback(
                "当前已有任务正在执行，"
                "拒绝新的执行任务"
            )
            return

        (
            valid,
            reason,
            normalized_steps,
            calculated_total,
        ) = self.validate_plan_steps(message)

        if not valid:
            self.publish_status(
                "FAILED",
                reason,
            )
            return

        self.task_id = (
            message.task_id.strip()
            or uuid.uuid4().hex[:12]
        )
        self.original_text = (
            message.original_text
        )
        self.steps = normalized_steps

        self.current_step_index = 0
        self.current_color = ""
        self.current_destination = ""
        self.target_classes = set()

        self.step_completed = 0
        self.step_total = 0
        self.total_completed = 0
        self.total_target = (
            calculated_total
        )

        self.locate_attempts = 0
        self.active = True

        self.publish_status(
            "PARSING",
            "任务计划校验成功，"
            f"共{len(self.steps)}个步骤，"
            f"总计{self.total_target}个物块",
        )

        self.start_current_step()

    def legacy_command_callback(
        self,
        message,
    ):
        if message.action == "stop":
            self.handle_stop(
                message.reason
            )
            return

        if message.action != "pick_and_place":
            self.publish_feedback(
                "旧任务命令动作不受支持"
            )
            return

        plan_message = TaskPlan()

        plan_message.header.stamp = (
            self.get_clock().now().to_msg()
        )
        plan_message.task_id = (
            "legacy_"
            + uuid.uuid4().hex[:8]
        )
        plan_message.original_text = (
            "旧TaskCommand兼容任务"
        )
        plan_message.action = "execute"
        plan_message.reason = ""

        step_message = TaskStep()
        step_message.color = message.color
        step_message.destination = (
            message.destination
        )
        step_message.count = message.count

        plan_message.steps = [
            step_message
        ]
        plan_message.total_count = (
            message.count
        )

        self.plan_callback(plan_message)

    def start_current_step(self):
        if (
            self.current_step_index
            >= len(self.steps)
        ):
            self.active = False

            self.publish_status(
                "FINISHED",
                "全部任务步骤执行完成，"
                f"共完成{self.total_completed}个物块",
                success=True,
            )
            return

        current_step = self.steps[
            self.current_step_index
        ]

        self.current_color = (
            current_step["color"]
        )
        self.current_destination = (
            current_step["destination"]
        )
        self.step_completed = 0
        self.step_total = (
            current_step["count"]
        )
        self.locate_attempts = 0

        self.target_classes = {
            f"{self.current_color}_block",
            f"{self.current_color}_cube",
        }

        self.publish_status(
            "LOCATING_OBJECT",
            "开始执行第"
            f"{self.current_step_index + 1}"
            f"/{len(self.steps)}步："
            f"寻找{self.current_color}物块，"
            f"目标区域{self.current_destination}，"
            f"数量{self.step_total}",
        )

    def target_is_visible(self):
        return bool(
            self.target_classes
            & self.visible_classes
        )

    def timer_callback(self):
        if not self.active:
            return

        if self.state == "LOCATING_OBJECT":
            if self.target_is_visible():
                self.locate_attempts = 0

                self.publish_status(
                    "NAVIGATING_TO_OBJECT",
                    f"已找到{self.current_color}"
                    "物块，模拟导航到目标",
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
                        "未找到第"
                        f"{self.current_step_index + 1}"
                        "步所需的"
                        f"{self.current_color}物块",
                    )
                else:
                    self.publish_status(
                        "LOCATING_OBJECT",
                        "正在寻找"
                        f"{self.current_color}物块，"
                        f"第{self.locate_attempts}次",
                    )

        elif (
            self.state
            == "NAVIGATING_TO_OBJECT"
        ):
            self.publish_status(
                "GRASPING",
                "模拟导航完成，正在抓取"
                f"{self.current_color}物块",
            )

        elif self.state == "GRASPING":
            self.publish_status(
                "NAVIGATING_TO_ZONE",
                "模拟抓取成功，前往"
                f"{self.current_destination}区",
            )

        elif (
            self.state
            == "NAVIGATING_TO_ZONE"
        ):
            self.publish_status(
                "PLACING",
                "已到达"
                f"{self.current_destination}区，"
                "正在放置",
            )

        elif self.state == "PLACING":
            self.step_completed += 1
            self.total_completed += 1

            if (
                self.step_completed
                < self.step_total
            ):
                self.locate_attempts = 0

                self.publish_status(
                    "LOCATING_OBJECT",
                    "第"
                    f"{self.current_step_index + 1}"
                    "步已完成"
                    f"{self.step_completed}/"
                    f"{self.step_total}，"
                    "开始寻找下一个"
                    f"{self.current_color}物块",
                )
            else:
                completed_step_number = (
                    self.current_step_index + 1
                )

                self.current_step_index += 1

                if (
                    self.current_step_index
                    < len(self.steps)
                ):
                    self.get_logger().info(
                        "第"
                        f"{completed_step_number}"
                        "步完成，进入下一步"
                    )

                    self.start_current_step()
                else:
                    self.active = False

                    self.publish_status(
                        "FINISHED",
                        "全部任务步骤执行完成，"
                        f"共完成"
                        f"{self.total_completed}个物块",
                        success=True,
                    )


def main(args=None):
    rclpy.init(args=args)

    node = TaskPlanManagerNode()

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
