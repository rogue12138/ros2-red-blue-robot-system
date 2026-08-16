#!/usr/bin/env python3

import cv2
import numpy as np
import rclpy

from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image

from robot_interfaces.msg import ObjectDetection
from robot_interfaces.msg import ObjectDetectionArray


COLOR_RANGES = {
    "red_cube": [
        (
            np.array([0, 90, 60]),
            np.array([12, 255, 255]),
        ),
        (
            np.array([168, 90, 60]),
            np.array([179, 255, 255]),
        ),
    ],
    "blue_cube": [
        (
            np.array([95, 80, 50]),
            np.array([135, 255, 255]),
        ),
    ],
}

DEBUG_COLORS = {
    "red_cube": (0, 0, 255),
    "blue_cube": (255, 0, 0),
}


class RedBlueDetectorNode(Node):

    def __init__(self):
        super().__init__(
            "red_blue_detector_node"
        )

        self.declare_parameter(
            "minimum_area",
            500.0,
        )
        self.declare_parameter(
            "maximum_area_ratio",
            0.20,
        )
        self.declare_parameter(
            "minimum_aspect_ratio",
            0.45,
        )
        self.declare_parameter(
            "maximum_aspect_ratio",
            2.20,
        )
        self.declare_parameter(
            "minimum_fill_ratio",
            0.50,
        )
        self.declare_parameter(
            "morph_kernel_size",
            5,
        )
        self.declare_parameter(
            "log_every_n_frames",
            30,
        )

        self.minimum_area = float(
            self.get_parameter(
                "minimum_area"
            ).value
        )
        self.maximum_area_ratio = float(
            self.get_parameter(
                "maximum_area_ratio"
            ).value
        )
        self.minimum_aspect_ratio = float(
            self.get_parameter(
                "minimum_aspect_ratio"
            ).value
        )
        self.maximum_aspect_ratio = float(
            self.get_parameter(
                "maximum_aspect_ratio"
            ).value
        )
        self.minimum_fill_ratio = float(
            self.get_parameter(
                "minimum_fill_ratio"
            ).value
        )
        self.morph_kernel_size = int(
            self.get_parameter(
                "morph_kernel_size"
            ).value
        )
        self.log_every_n_frames = int(
            self.get_parameter(
                "log_every_n_frames"
            ).value
        )

        if self.morph_kernel_size < 1:
            self.morph_kernel_size = 1

        if self.morph_kernel_size % 2 == 0:
            self.morph_kernel_size += 1

        if self.log_every_n_frames < 1:
            self.log_every_n_frames = 1

        self.bridge = CvBridge()
        self.frame_count = 0

        self.image_subscription = (
            self.create_subscription(
                Image,
                "/camera/image_raw",
                self.image_callback,
                10,
            )
        )

        self.detection_publisher = (
            self.create_publisher(
                ObjectDetectionArray,
                "/vision/detections",
                10,
            )
        )

        self.debug_image_publisher = (
            self.create_publisher(
                Image,
                "/vision/debug_image",
                10,
            )
        )

        self.get_logger().info(
            "正式红蓝立方体检测节点已启动"
        )

    def create_color_mask(
        self,
        hsv_image,
        ranges,
    ):
        mask = np.zeros(
            hsv_image.shape[:2],
            dtype=np.uint8,
        )

        for lower, upper in ranges:
            current_mask = cv2.inRange(
                hsv_image,
                lower,
                upper,
            )

            mask = cv2.bitwise_or(
                mask,
                current_mask,
            )

        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (
                self.morph_kernel_size,
                self.morph_kernel_size,
            ),
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            kernel,
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            kernel,
        )

        return mask

    def calculate_confidence(
        self,
        area,
        bounding_area,
        image_area,
    ):
        if bounding_area <= 0:
            return 0.0

        fill_ratio = area / bounding_area
        area_ratio = area / image_area

        fill_score = min(
            1.0,
            max(0.0, fill_ratio),
        )

        area_score = min(
            1.0,
            area_ratio / 0.02,
        )

        confidence = (
            0.70 * fill_score
            + 0.30 * area_score
        )

        return float(
            min(0.99, max(0.0, confidence))
        )

    def detect_color_objects(
        self,
        hsv_image,
        class_name,
        ranges,
    ):
        mask = self.create_color_mask(
            hsv_image,
            ranges,
        )

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        image_height, image_width = (
            hsv_image.shape[:2]
        )
        image_area = float(
            image_height * image_width
        )
        maximum_area = (
            image_area
            * self.maximum_area_ratio
        )

        detections = []

        for contour in contours:
            area = float(
                cv2.contourArea(contour)
            )

            if area < self.minimum_area:
                continue

            if area > maximum_area:
                continue

            x, y, width, height = (
                cv2.boundingRect(contour)
            )

            if height <= 0 or width <= 0:
                continue

            aspect_ratio = (
                float(width) / float(height)
            )

            if (
                aspect_ratio
                < self.minimum_aspect_ratio
            ):
                continue

            if (
                aspect_ratio
                > self.maximum_aspect_ratio
            ):
                continue

            bounding_area = float(
                width * height
            )
            fill_ratio = (
                area / bounding_area
            )

            if (
                fill_ratio
                < self.minimum_fill_ratio
            ):
                continue

            detection = ObjectDetection()

            detection.class_name = class_name
            detection.confidence = (
                self.calculate_confidence(
                    area,
                    bounding_area,
                    image_area,
                )
            )

            detection.x_min = int(x)
            detection.y_min = int(y)
            detection.x_max = int(
                x + width
            )
            detection.y_max = int(
                y + height
            )

            detection.has_position = False
            detection.position.x = 0.0
            detection.position.y = 0.0
            detection.position.z = 0.0

            detections.append(detection)

        detections.sort(
            key=lambda item: (
                item.x_min,
                item.y_min,
            )
        )

        return detections

    def draw_detection(
        self,
        image,
        detection,
        detection_id,
    ):
        color = DEBUG_COLORS[
            detection.class_name
        ]

        center_x = int(
            (
                detection.x_min
                + detection.x_max
            )
            / 2
        )
        center_y = int(
            (
                detection.y_min
                + detection.y_max
            )
            / 2
        )

        cv2.rectangle(
            image,
            (
                detection.x_min,
                detection.y_min,
            ),
            (
                detection.x_max,
                detection.y_max,
            ),
            color,
            2,
        )

        cv2.drawMarker(
            image,
            (
                center_x,
                center_y,
            ),
            color,
            markerType=cv2.MARKER_CROSS,
            markerSize=12,
            thickness=2,
        )

        label = (
            f"ID:{detection_id} "
            f"{detection.class_name} "
            f"{detection.confidence:.2f} "
            f"px:({center_x},{center_y})"
        )

        text_y = max(
            20,
            detection.y_min - 8,
        )

        cv2.putText(
            image,
            label,
            (
                detection.x_min,
                text_y,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            2,
            cv2.LINE_AA,
        )
    def image_callback(self, message):
        try:
            image = (
                self.bridge.imgmsg_to_cv2(
                    message,
                    desired_encoding="bgr8",
                )
            )
        except Exception as error:
            self.get_logger().error(
                f"图像转换失败：{error}"
            )
            return

        hsv_image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2HSV,
        )

        all_detections = []

        for class_name, ranges in (
            COLOR_RANGES.items()
        ):
            color_detections = (
                self.detect_color_objects(
                    hsv_image,
                    class_name,
                    ranges,
                )
            )

            all_detections.extend(
                color_detections
            )
        all_detections.sort(
            key=lambda item: (
                item.x_min,
                item.y_min,
                item.class_name,
            )
        )
        detection_array = (
            ObjectDetectionArray()
        )
        detection_array.header = (
            message.header
        )
        detection_array.detections = (
            all_detections
        )

        self.detection_publisher.publish(
            detection_array
        )

        debug_image = image.copy()

        for detection_id, detection in enumerate(
            all_detections,
            start=1,
        ):
            self.draw_detection(
                debug_image,
                detection,
                detection_id,
            )
        try:
            debug_message = (
                self.bridge.cv2_to_imgmsg(
                    debug_image,
                    encoding="bgr8",
                )
            )
            debug_message.header = (
                message.header
            )

            self.debug_image_publisher.publish(
                debug_message
            )
        except Exception as error:
            self.get_logger().error(
                f"调试图像发布失败：{error}"
            )

        self.frame_count += 1

        if (
            self.frame_count
            % self.log_every_n_frames
            == 0
        ):
            red_count = sum(
                detection.class_name
                == "red_cube"
                for detection in all_detections
            )
            blue_count = sum(
                detection.class_name
                == "blue_cube"
                for detection in all_detections
            )

            self.get_logger().info(
                "当前检测结果："
                f"red_cube={red_count}，"
                f"blue_cube={blue_count}"
            )


def main(args=None):
    rclpy.init(args=args)

    node = RedBlueDetectorNode()

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
