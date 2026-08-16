#!/usr/bin/env python3

import json

import numpy as np
import rclpy

from rclpy.node import Node
from std_msgs.msg import String

from robot_interfaces.msg import ObjectDetectionArray


TARGET_CLASSES = (
    "red_cube",
    "blue_cube",
)


class PositionStabilityAcceptanceNode(Node):

    def __init__(self):
        super().__init__(
            "position_stability_acceptance_node"
        )

        self.declare_parameter(
            "input_topic",
            "/vision/localized_detections",
        )
        self.declare_parameter(
            "result_topic",
            "/vision/stability_result",
        )
        self.declare_parameter(
            "required_samples_per_class",
            30,
        )
        self.declare_parameter(
            "expected_filter_window",
            5,
        )
        self.declare_parameter(
            "maximum_axis_std_m",
            0.01,
        )
        self.declare_parameter(
            "maximum_axis_range_m",
            0.03,
        )

        self.input_topic = str(
            self.get_parameter(
                "input_topic"
            ).value
        )
        self.result_topic = str(
            self.get_parameter(
                "result_topic"
            ).value
        )
        self.required_samples = int(
            self.get_parameter(
                "required_samples_per_class"
            ).value
        )
        self.expected_filter_window = int(
            self.get_parameter(
                "expected_filter_window"
            ).value
        )
        self.maximum_axis_std_m = float(
            self.get_parameter(
                "maximum_axis_std_m"
            ).value
        )
        self.maximum_axis_range_m = float(
            self.get_parameter(
                "maximum_axis_range_m"
            ).value
        )

        if self.required_samples < 5:
            raise ValueError(
                "required_samples_per_class不能小于5"
            )

        if self.expected_filter_window < 1:
            raise ValueError(
                "expected_filter_window必须大于0"
            )

        if self.maximum_axis_std_m <= 0.0:
            raise ValueError(
                "maximum_axis_std_m必须大于0"
            )

        if self.maximum_axis_range_m <= 0.0:
            raise ValueError(
                "maximum_axis_range_m必须大于0"
            )

        self.samples = {
            class_name: []
            for class_name in TARGET_CLASSES
        }

        self.output_frame = ""
        self.finished = False

        self.result_publisher = (
            self.create_publisher(
                String,
                self.result_topic,
                10,
            )
        )

        self.detection_subscription = (
            self.create_subscription(
                ObjectDetectionArray,
                self.input_topic,
                self.detection_callback,
                10,
            )
        )

        self.get_logger().info(
            "位置稳定性验收节点已启动："
            f"每类采集{self.required_samples}个位置，"
            f"标准差阈值={self.maximum_axis_std_m:.3f}m，"
            f"极差阈值={self.maximum_axis_range_m:.3f}m"
        )

    def collect_detection(
        self,
        detection,
    ):
        class_name = detection.class_name

        if class_name not in self.samples:
            return

        if not detection.has_position:
            return

        if (
            len(self.samples[class_name])
            >= self.required_samples
        ):
            return

        position = np.asarray(
            [
                float(detection.position.x),
                float(detection.position.y),
                float(detection.position.z),
            ],
            dtype=np.float64,
        )

        if not np.all(np.isfinite(position)):
            return

        self.samples[class_name].append(
            position
        )

        sample_count = len(
            self.samples[class_name]
        )

        if (
            sample_count == 1
            or sample_count % 5 == 0
        ):
            self.get_logger().info(
                f"{class_name}稳定性采样："
                f"{sample_count}/"
                f"{self.required_samples}"
            )

    def calculate_class_result(
        self,
        class_name,
    ):
        position_array = np.asarray(
            self.samples[class_name],
            dtype=np.float64,
        )

        mean_position = np.mean(
            position_array,
            axis=0,
        )
        axis_std = np.std(
            position_array,
            axis=0,
        )
        axis_min = np.min(
            position_array,
            axis=0,
        )
        axis_max = np.max(
            position_array,
            axis=0,
        )
        axis_range = axis_max - axis_min

        maximum_std = float(
            np.max(axis_std)
        )
        maximum_range = float(
            np.max(axis_range)
        )

        class_pass = (
            maximum_std
            <= self.maximum_axis_std_m
            and maximum_range
            <= self.maximum_axis_range_m
        )

        return {
            "samples": int(
                position_array.shape[0]
            ),
            "mean_position_m": {
                "x": float(mean_position[0]),
                "y": float(mean_position[1]),
                "z": float(mean_position[2]),
            },
            "axis_std_m": {
                "x": float(axis_std[0]),
                "y": float(axis_std[1]),
                "z": float(axis_std[2]),
            },
            "axis_range_m": {
                "x": float(axis_range[0]),
                "y": float(axis_range[1]),
                "z": float(axis_range[2]),
            },
            "maximum_axis_std_m": maximum_std,
            "maximum_axis_range_m": maximum_range,
            "pass": class_pass,
        }

    def publish_result(self):
        class_results = {}
        overall_pass = True

        for class_name in TARGET_CLASSES:
            class_result = (
                self.calculate_class_result(
                    class_name
                )
            )

            class_results[class_name] = (
                class_result
            )
            overall_pass = (
                overall_pass
                and class_result["pass"]
            )

        result = {
            "test_environment": (
                "fixed_objects_with_mock_depth"
            ),
            "output_frame": self.output_frame,
            "expected_filter_window": (
                self.expected_filter_window
            ),
            "required_samples_per_class": (
                self.required_samples
            ),
            "maximum_allowed_axis_std_m": (
                self.maximum_axis_std_m
            ),
            "maximum_allowed_axis_range_m": (
                self.maximum_axis_range_m
            ),
            "classes": class_results,
            "overall_result": (
                "PASS"
                if overall_pass
                else "FAIL"
            ),
        }

        result_text = json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        )

        message = String()
        message.data = result_text

        self.result_publisher.publish(message)

        self.get_logger().info(
            "位置稳定性验收完成：\n"
            f"{result_text}"
        )

    def detection_callback(self, message):
        if self.finished:
            return

        if message.header.frame_id:
            self.output_frame = (
                message.header.frame_id
            )

        for detection in message.detections:
            self.collect_detection(
                detection
            )

        all_finished = all(
            len(self.samples[class_name])
            >= self.required_samples
            for class_name in TARGET_CLASSES
        )

        if all_finished:
            self.finished = True
            self.publish_result()


def main(args=None):
    rclpy.init(args=args)

    node = PositionStabilityAcceptanceNode()

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
