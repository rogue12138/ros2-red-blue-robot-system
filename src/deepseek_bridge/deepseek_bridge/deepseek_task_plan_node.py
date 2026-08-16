#!/usr/bin/env python3

import json
import os
import uuid
from pathlib import Path

import requests
import rclpy

from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from std_msgs.msg import String

from robot_interfaces.msg import TaskPlan
from robot_interfaces.msg import TaskStep


ALLOWED_ACTIONS = {
    "execute",
    "stop",
    "clarify",
    "reject",
}

ALLOWED_COLORS = {
    "red",
    "blue",
}

ALLOWED_DESTINATIONS = {
    "A",
    "B",
    "C",
}

STOP_KEYWORDS = {
    "停止",
    "立即停止",
    "紧急停止",
    "急停",
    "终止任务",
    "取消任务",
    "取消当前任务",
    "不要继续",
}

DANGEROUS_KEYWORDS = {
    "关闭避障",
    "禁用避障",
    "绕过避障",
    "关闭雷达",
    "禁用雷达",
    "绕过雷达",
    "关闭安全",
    "禁用安全",
    "绕过安全",
    "碰撞障碍物",
    "全速撞",
}


class DeepSeekTaskPlanNode(Node):

    def __init__(self):
        super().__init__("deepseek_task_plan_node")

        self.declare_parameter(
            "api_url",
            "https://api.deepseek.com/chat/completions",
        )
        self.declare_parameter(
            "model",
            "deepseek-v4-flash",
        )
        self.declare_parameter(
            "timeout_sec",
            30.0,
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
            "prompt_file",
            "",
        )

        self.api_url = str(
            self.get_parameter("api_url").value
        )
        self.model = str(
            self.get_parameter("model").value
        )
        self.timeout_sec = float(
            self.get_parameter("timeout_sec").value
        )
        self.max_step_count = int(
            self.get_parameter("max_step_count").value
        )
        self.max_total_count = int(
            self.get_parameter("max_total_count").value
        )

        self.api_key = os.environ.get("DEEPSEEK_API_KEY")

        self.system_prompt = self.load_system_prompt()

        self.command_subscription = self.create_subscription(
            String,
            "/user/text_command",
            self.command_callback,
            10,
        )

        self.feedback_publisher = self.create_publisher(
            String,
            "/llm/feedback",
            10,
        )

        self.task_plan_publisher = self.create_publisher(
            TaskPlan,
            "/llm/task_plan",
            10,
        )

        self.get_logger().info(
            "DeepSeek多步骤任务解析节点已启动"
        )

        if self.api_key:
            self.get_logger().info(
                "DeepSeek API密钥已从环境变量加载"
            )
        else:
            self.get_logger().error(
                "没有找到DEEPSEEK_API_KEY环境变量"
            )

    def load_system_prompt(self):
        configured_path = str(
            self.get_parameter("prompt_file").value
        ).strip()

        if configured_path:
            prompt_path = Path(
                configured_path
            ).expanduser()
        else:
            package_share = Path(
                get_package_share_directory(
                    "deepseek_bridge"
                )
            )
            prompt_path = (
                package_share
                / "config"
                / "task_plan_prompt.txt"
            )

        if not prompt_path.is_file():
            raise FileNotFoundError(
                f"找不到任务提示词文件：{prompt_path}"
            )

        prompt_text = prompt_path.read_text(
            encoding="utf-8"
        ).strip()

        if not prompt_text:
            raise ValueError("任务提示词文件为空")

        self.get_logger().info(
            f"已加载任务提示词：{prompt_path}"
        )

        return prompt_text

    def publish_feedback(self, text):
        message = String()
        message.data = text
        self.feedback_publisher.publish(message)

    def publish_task_plan(
        self,
        parsed_plan,
        original_text,
    ):
        message = TaskPlan()

        message.header.stamp = (
            self.get_clock().now().to_msg()
        )
        message.task_id = uuid.uuid4().hex[:12]
        message.original_text = original_text
        message.action = parsed_plan["action"]
        message.reason = parsed_plan["reason"]

        message.steps = []

        total_count = 0

        for step_data in parsed_plan["steps"]:
            step_message = TaskStep()
            step_message.color = step_data["color"]
            step_message.destination = (
                step_data["destination"]
            )
            step_message.count = step_data["count"]

            message.steps.append(step_message)
            total_count += step_message.count

        message.total_count = total_count

        self.task_plan_publisher.publish(message)

        self.get_logger().info(
            "已发布任务计划："
            f"task_id={message.task_id}, "
            f"action={message.action}, "
            f"steps={len(message.steps)}, "
            f"total_count={message.total_count}"
        )

    def call_deepseek(self, user_text):
        if not self.api_key:
            raise RuntimeError(
                "没有设置DEEPSEEK_API_KEY环境变量"
            )

        headers = {
            "Authorization": (
                f"Bearer {self.api_key}"
            ),
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": self.system_prompt,
                },
                {
                    "role": "user",
                    "content": user_text,
                },
            ],
            "response_format": {
                "type": "json_object",
            },
            "thinking": {
                "type": "disabled",
            },
            "temperature": 0,
            "max_tokens": 800,
        }

        response = requests.post(
            self.api_url,
            headers=headers,
            json=payload,
            timeout=self.timeout_sec,
        )

        response.raise_for_status()

        result = response.json()

        try:
            content = (
                result["choices"][0]
                ["message"]["content"]
            )
        except (
            KeyError,
            IndexError,
            TypeError,
        ) as error:
            raise ValueError(
                "DeepSeek响应中缺少对话内容"
            ) from error

        if not content:
            raise ValueError(
                "DeepSeek返回了空内容"
            )

        return json.loads(content)

    def validate_plan(self, plan):
        if not isinstance(plan, dict):
            raise ValueError(
                "返回结果不是JSON对象"
            )

        required_fields = {
            "action",
            "steps",
            "reason",
        }

        missing_fields = (
            required_fields - plan.keys()
        )

        if missing_fields:
            raise ValueError(
                f"缺少字段：{sorted(missing_fields)}"
            )

        action = plan["action"]

        if not isinstance(action, str):
            raise ValueError(
                "action必须是字符串"
            )

        action = action.strip().lower()

        if action not in ALLOWED_ACTIONS:
            raise ValueError(
                f"不允许的动作：{action}"
            )

        reason = plan["reason"]

        if not isinstance(reason, str):
            raise ValueError(
                "reason必须是字符串"
            )

        steps = plan["steps"]

        if not isinstance(steps, list):
            raise ValueError(
                "steps必须是数组"
            )

        if action != "execute":
            if not reason.strip():
                default_reasons = {
                    "stop": "用户要求停止当前任务",
                    "clarify": "任务信息不完整",
                    "reject": "任务不符合安全规则",
                }
                reason = default_reasons[action]

            return {
                "action": action,
                "steps": [],
                "reason": reason.strip(),
            }

        if not steps:
            raise ValueError(
                "execute任务的steps不能为空"
            )

        if len(steps) > self.max_step_count:
            raise ValueError(
                "任务步骤数量超过限制："
                f"{len(steps)} > "
                f"{self.max_step_count}"
            )

        normalized_steps = []
        total_count = 0

        for index, step in enumerate(
            steps,
            start=1,
        ):
            if not isinstance(step, dict):
                raise ValueError(
                    f"第{index}个步骤不是JSON对象"
                )

            step_required_fields = {
                "color",
                "destination",
                "count",
            }

            step_missing_fields = (
                step_required_fields
                - step.keys()
            )

            if step_missing_fields:
                raise ValueError(
                    f"第{index}个步骤缺少字段："
                    f"{sorted(step_missing_fields)}"
                )

            color = step["color"]
            destination = step["destination"]
            count = step["count"]

            if not isinstance(color, str):
                raise ValueError(
                    f"第{index}个步骤的color"
                    "必须是字符串"
                )

            color = color.strip().lower()

            if color not in ALLOWED_COLORS:
                raise ValueError(
                    f"第{index}个步骤包含"
                    f"不允许的颜色：{color}"
                )

            if not isinstance(
                destination,
                str,
            ):
                raise ValueError(
                    f"第{index}个步骤的"
                    "destination必须是字符串"
                )

            destination = (
                destination.strip().upper()
            )

            if (
                destination
                not in ALLOWED_DESTINATIONS
            ):
                raise ValueError(
                    f"第{index}个步骤包含"
                    f"不允许的区域："
                    f"{destination}"
                )

            if type(count) is not int:
                raise ValueError(
                    f"第{index}个步骤的"
                    "count必须是整数"
                )

            if not 1 <= count <= 8:
                raise ValueError(
                    f"第{index}个步骤的"
                    "count必须在1到8之间"
                )

            total_count += count

            normalized_steps.append(
                {
                    "color": color,
                    "destination": destination,
                    "count": count,
                }
            )

        if total_count > self.max_total_count:
            raise ValueError(
                "任务总数量超过限制："
                f"{total_count} > "
                f"{self.max_total_count}"
            )

        return {
            "action": "execute",
            "steps": normalized_steps,
            "reason": reason.strip(),
        }

    def make_local_stop_plan(self):
        return {
            "action": "stop",
            "steps": [],
            "reason": "用户要求立即停止当前任务",
        }

    def make_local_reject_plan(self):
        return {
            "action": "reject",
            "steps": [],
            "reason": "请求涉及绕过机器人安全系统",
        }

    def contains_keyword(
        self,
        user_text,
        keywords,
    ):
        return any(
            keyword in user_text
            for keyword in keywords
        )

    def command_callback(self, message):
        user_text = message.data.strip()

        if not user_text:
            self.get_logger().warning(
                "收到空指令"
            )
            self.publish_feedback(
                "指令为空，请重新输入"
            )
            return

        self.get_logger().info(
            f"正在解析用户指令：{user_text}"
        )

        try:
            if self.contains_keyword(
                user_text,
                STOP_KEYWORDS,
            ):
                plan = self.make_local_stop_plan()

            elif self.contains_keyword(
                user_text,
                DANGEROUS_KEYWORDS,
            ):
                plan = self.make_local_reject_plan()

            else:
                plan = self.call_deepseek(
                    user_text
                )
                plan = self.validate_plan(
                    plan
                )

            plan_text = json.dumps(
                plan,
                ensure_ascii=False,
            )

            self.get_logger().info(
                f"解析成功：{plan_text}"
            )

            self.publish_task_plan(
                plan,
                user_text,
            )

            if plan["action"] == "execute":
                self.publish_feedback(
                    plan_text
                )

            elif plan["action"] == "stop":
                self.publish_feedback(
                    "已请求紧急停止当前任务"
                )

            elif plan["action"] == "clarify":
                self.publish_feedback(
                    "请补充任务信息："
                    f"{plan['reason']}"
                )

            elif plan["action"] == "reject":
                self.publish_feedback(
                    "任务已拒绝："
                    f"{plan['reason']}"
                )

        except requests.exceptions.Timeout:
            self.get_logger().error(
                "DeepSeek API请求超时"
            )
            self.publish_feedback(
                "语言服务请求超时，任务未执行"
            )

        except requests.exceptions.RequestException as error:
            self.get_logger().error(
                f"DeepSeek API请求失败：{error}"
            )
            self.publish_feedback(
                "语言服务请求失败，任务未执行"
            )

        except (
            KeyError,
            IndexError,
            TypeError,
            ValueError,
            RuntimeError,
            json.JSONDecodeError,
        ) as error:
            self.get_logger().error(
                f"返回结果校验失败：{error}"
            )
            self.publish_feedback(
                "指令解析结果不合法，任务未执行："
                f"{error}"
            )


def main(args=None):
    rclpy.init(args=args)

    node = DeepSeekTaskPlanNode()

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
