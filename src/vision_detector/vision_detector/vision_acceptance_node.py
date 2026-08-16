#!/usr/bin/env python3

import json

import rclpy

from message_filters import ApproximateTimeSynchronizer
from message_filters import Subscriber
from rclpy.node import Node
from std_msgs.msg import String

from robot_interfaces.msg import ObjectDetectionArray


TARGET_CLASSES = (
    "red_cube",
    "blue_cube",
)


class VisionAcceptanceNode(Node):

    def __init__(self):
        super().__init__("vision_acceptance_node")

        self.declare_parameter(
            "prediction_topic",
            "/vision/detections",
        )
        self.declare_parameter(
            "ground_truth_topic",
            "/simulation/ground_truth_detections",
        )
        self.declare_parameter(
            "result_topic",
            "/vision/acceptance_result",
        )

        self.declare_parameter(
            "required_trials_per_class",
            10,
        )
        self.declare_parameter(
            "minimum_iou",
            0.50,
        )
        self.declare_parameter(
            "edge_tolerance_pixels",
            10,
        )
        self.declare_parameter(
            "minimum_confidence",
            0.20,
        )
        self.declare_parameter(
            "synchronization_queue_size",
            30,
        )
        self.declare_parameter(
            "synchronization_slop_sec",
            0.10,
        )

        self.prediction_topic = str(
            self.get_parameter(
                "prediction_topic"
            ).value
        )
        self.ground_truth_topic = str(
            self.get_parameter(
                "ground_truth_topic"
            ).value
        )
        self.result_topic = str(
            self.get_parameter(
                "result_topic"
            ).value
        )

        self.required_trials = int(
            self.get_parameter(
                "required_trials_per_class"
            ).value
        )
        self.minimum_iou = float(
            self.get_parameter(
                "minimum_iou"
            ).value
        )
        self.edge_tolerance_pixels = int(
            self.get_parameter(
                "edge_tolerance_pixels"
            ).value
        )
        self.minimum_confidence = float(
            self.get_parameter(
                "minimum_confidence"
            ).value
        )

        queue_size = int(
            self.get_parameter(
                "synchronization_queue_size"
            ).value
        )
        synchronization_slop = float(
            self.get_parameter(
                "synchronization_slop_sec"
            ).value
        )

        if self.required_trials < 1:
            raise ValueError(
                "required_trials_per_class必须大于0"
            )

        if not 0.0 <= self.minimum_iou <= 1.0:
            raise ValueError(
                "minimum_iou必须在0到1之间"
            )

        if self.edge_tolerance_pixels < 0:
            raise ValueError(
                "edge_tolerance_pixels不能小于0"
            )

        if not 0.0 <= self.minimum_confidence <= 1.0:
            raise ValueError(
                "minimum_confidence必须在0到1之间"
            )

        if queue_size < 1:
            queue_size = 1

        if synchronization_slop < 0.0:
            synchronization_slop = 0.0

        self.statistics = {}

        for class_name in TARGET_CLASSES:
            self.statistics[class_name] = {
                "trials": 0,
                "detection_successes": 0,
                "edge_successes": 0,
                "edge_errors": [],
                "maximum_edge_error": 0,
            }

        self.last_scene_signature = None
        self.finished = False

        self.result_publisher = (
            self.create_publisher(
                String,
                self.result_topic,
                10,
            )
        )

        self.prediction_subscriber = Subscriber(
            self,
            ObjectDetectionArray,
            self.prediction_topic,
            qos_profile=10,
        )

        self.ground_truth_subscriber = Subscriber(
            self,
            ObjectDetectionArray,
            self.ground_truth_topic,
            qos_profile=10,
        )

        self.synchronizer = (
            ApproximateTimeSynchronizer(
                [
                    self.prediction_subscriber,
                    self.ground_truth_subscriber,
                ],
                queue_size=queue_size,
                slop=synchronization_slop,
            )
        )

        self.synchronizer.registerCallback(
            self.synchronized_callback
        )

        self.get_logger().info(
            "二维视觉自动验收节点已启动："
            f"每类测试{self.required_trials}个场景，"
            f"边缘容差={self.edge_tolerance_pixels}px，"
            f"最低IoU={self.minimum_iou:.2f}"
        )

    def create_scene_signature(
        self,
        ground_truth_message,
    ):
        values = []

        for detection in (
            ground_truth_message.detections
        ):
            if detection.class_name not in TARGET_CLASSES:
                continue

            values.append(
                (
                    detection.class_name,
                    int(detection.x_min),
                    int(detection.y_min),
                    int(detection.x_max),
                    int(detection.y_max),
                )
            )

        values.sort()

        return tuple(values)

    def calculate_iou(
        self,
        first,
        second,
    ):
        intersection_x_min = max(
            int(first.x_min),
            int(second.x_min),
        )
        intersection_y_min = max(
            int(first.y_min),
            int(second.y_min),
        )
        intersection_x_max = min(
            int(first.x_max),
            int(second.x_max),
        )
        intersection_y_max = min(
            int(first.y_max),
            int(second.y_max),
        )

        intersection_width = max(
            0,
            intersection_x_max
            - intersection_x_min,
        )
        intersection_height = max(
            0,
            intersection_y_max
            - intersection_y_min,
        )

        intersection_area = (
            intersection_width
            * intersection_height
        )

        first_area = max(
            0,
            int(first.x_max)
            - int(first.x_min),
        ) * max(
            0,
            int(first.y_max)
            - int(first.y_min),
        )

        second_area = max(
            0,
            int(second.x_max)
            - int(second.x_min),
        ) * max(
            0,
            int(second.y_max)
            - int(second.y_min),
        )

        union_area = (
            first_area
            + second_area
            - intersection_area
        )

        if union_area <= 0:
            return 0.0

        return float(
            intersection_area / union_area
        )

    def calculate_edge_errors(
        self,
        prediction,
        ground_truth,
    ):
        return (
            abs(
                int(prediction.x_min)
                - int(ground_truth.x_min)
            ),
            abs(
                int(prediction.y_min)
                - int(ground_truth.y_min)
            ),
            abs(
                int(prediction.x_max)
                - int(ground_truth.x_max)
            ),
            abs(
                int(prediction.y_max)
                - int(ground_truth.y_max)
            ),
        )

    def match_detections(
        self,
        predictions,
        ground_truths,
    ):
        matches = []
        used_prediction_indices = set()

        for ground_truth in ground_truths:
            best_index = None
            best_iou = -1.0

            for index, prediction in enumerate(
                predictions
            ):
                if index in used_prediction_indices:
                    continue

                current_iou = self.calculate_iou(
                    prediction,
                    ground_truth,
                )

                if current_iou > best_iou:
                    best_iou = current_iou
                    best_index = index

            if best_index is None:
                continue

            used_prediction_indices.add(
                best_index
            )

            matches.append(
                (
                    predictions[best_index],
                    ground_truth,
                    best_iou,
                )
            )

        return matches

    def evaluate_class(
        self,
        class_name,
        prediction_message,
        ground_truth_message,
    ):
        statistic = self.statistics[class_name]

        if statistic["trials"] >= self.required_trials:
            return

        ground_truths = [
            detection
            for detection in ground_truth_message.detections
            if detection.class_name == class_name
        ]

        if not ground_truths:
            return

        predictions = [
            detection
            for detection in prediction_message.detections
            if (
                detection.class_name == class_name
                and detection.confidence
                >= self.minimum_confidence
            )
        ]

        matches = self.match_detections(
            predictions,
            ground_truths,
        )

        exact_count = (
            len(predictions) == len(ground_truths)
        )
        every_target_matched = (
            len(matches) == len(ground_truths)
        )
        iou_passed = (
            every_target_matched
            and all(
                match[2] >= self.minimum_iou
                for match in matches
            )
        )

        detection_success = (
            exact_count
            and iou_passed
        )

        edge_errors = []

        for prediction, ground_truth, _ in matches:
            edge_errors.extend(
                self.calculate_edge_errors(
                    prediction,
                    ground_truth,
                )
            )

        maximum_scene_edge_error = (
            max(edge_errors)
            if edge_errors
            else None
        )

        edge_success = (
            detection_success
            and maximum_scene_edge_error is not None
            and maximum_scene_edge_error
            <= self.edge_tolerance_pixels
        )

        statistic["trials"] += 1

        if detection_success:
            statistic["detection_successes"] += 1

        if edge_success:
            statistic["edge_successes"] += 1

        if edge_errors:
            statistic["edge_errors"].extend(
                edge_errors
            )
            statistic["maximum_edge_error"] = max(
                statistic["maximum_edge_error"],
                maximum_scene_edge_error,
            )

        self.get_logger().info(
            f"{class_name}验收"
            f"[{statistic['trials']}/"
            f"{self.required_trials}]："
            f"真实数量={len(ground_truths)}，"
            f"检测数量={len(predictions)}，"
            f"检测={'PASS' if detection_success else 'FAIL'}，"
            "最大边缘误差="
            f"{maximum_scene_edge_error}px"
        )

    def create_result(self):
        result = {
            "required_trials_per_class": (
                self.required_trials
            ),
            "minimum_required_rate": 0.95,
            "minimum_iou": self.minimum_iou,
            "edge_tolerance_pixels": (
                self.edge_tolerance_pixels
            ),
            "classes": {},
        }

        overall_pass = True

        for class_name in TARGET_CLASSES:
            statistic = self.statistics[class_name]

            trials = statistic["trials"]
            detection_rate = (
                statistic["detection_successes"]
                / trials
                if trials > 0
                else 0.0
            )
            edge_rate = (
                statistic["edge_successes"]
                / trials
                if trials > 0
                else 0.0
            )

            mean_edge_error = (
                sum(statistic["edge_errors"])
                / len(statistic["edge_errors"])
                if statistic["edge_errors"]
                else None
            )

            class_pass = (
                trials >= self.required_trials
                and detection_rate >= 0.95
                and edge_rate >= 0.95
                and statistic["maximum_edge_error"]
                <= self.edge_tolerance_pixels
            )

            result["classes"][class_name] = {
                "trials": trials,
                "detection_successes": (
                    statistic["detection_successes"]
                ),
                "detection_rate": detection_rate,
                "edge_successes": (
                    statistic["edge_successes"]
                ),
                "edge_rate": edge_rate,
                "mean_edge_error_pixels": (
                    mean_edge_error
                ),
                "maximum_edge_error_pixels": (
                    statistic["maximum_edge_error"]
                ),
                "pass": class_pass,
            }

            overall_pass = (
                overall_pass and class_pass
            )

        result["overall_result"] = (
            "PASS" if overall_pass else "FAIL"
        )

        return result

    def publish_final_result(self):
        result = self.create_result()

        result_text = json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        )

        message = String()
        message.data = result_text

        self.result_publisher.publish(message)

        self.get_logger().info(
            "视觉自动验收完成：\n"
            f"{result_text}"
        )

    def synchronized_callback(
        self,
        prediction_message,
        ground_truth_message,
    ):
        if self.finished:
            return

        scene_signature = (
            self.create_scene_signature(
                ground_truth_message
            )
        )

        if not scene_signature:
            return

        if (
            scene_signature
            == self.last_scene_signature
        ):
            return

        self.last_scene_signature = scene_signature

        for class_name in TARGET_CLASSES:
            self.evaluate_class(
                class_name,
                prediction_message,
                ground_truth_message,
            )

        all_finished = all(
            self.statistics[class_name]["trials"]
            >= self.required_trials
            for class_name in TARGET_CLASSES
        )

        if all_finished:
            self.finished = True
            self.publish_final_result()


def main(args=None):
    rclpy.init(args=args)

    node = VisionAcceptanceNode()

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
