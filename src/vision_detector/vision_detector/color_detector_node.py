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
    "red_block": [
        (
            np.array([0, 100, 100]),
            np.array([10, 255, 255]),
        ),
        (
            np.array([170, 100, 100]),
            np.array([179, 255, 255]),
        ),
    ],
    "green_block": [
        (
            np.array([35, 80, 80]),
            np.array([85, 255, 255]),
        ),
    ],
    "blue_block": [
        (
            np.array([90, 80, 80]),
            np.array([130, 255, 255]),
        ),
    ],
    "yellow_block": [
        (
            np.array([20, 100, 100]),
            np.array([35, 255, 255]),
        ),
    ],
}


class ColorDetectorNode(Node):

    def __init__(self):
        super().__init__("color_detector_node")

        self.declare_parameter("minimum_area", 500.0)

        self.minimum_area = (
            self.get_parameter("minimum_area")
            .get_parameter_value()
            .double_value
        )

        self.bridge = CvBridge()

        self.image_subscription = self.create_subscription(
            Image,
            "/camera/image_raw",
            self.image_callback,
            10,
        )

        self.detection_publisher = self.create_publisher(
            ObjectDetectionArray,
            "/vision/detections",
            10,
        )
        self.debug_image_publisher = self.create_publisher(
            Image,
            "/vision/debug_image",
            10,
        )

        self.frame_count = 0

        self.get_logger().info(
            "HSV颜色检测节点已启动"
        )

    def create_color_mask(self, hsv_image, ranges):
        combined_mask = np.zeros(
            hsv_image.shape[:2],
            dtype=np.uint8,
        )

        for lower, upper in ranges:
            mask = cv2.inRange(
                hsv_image,
                lower,
                upper,
            )
            combined_mask = cv2.bitwise_or(
                combined_mask,
                mask,
            )

        kernel = np.ones(
            (5, 5),
            dtype=np.uint8,
        )

        combined_mask = cv2.morphologyEx(
            combined_mask,
            cv2.MORPH_OPEN,
            kernel,
        )

        combined_mask = cv2.morphologyEx(
            combined_mask,
            cv2.MORPH_CLOSE,
            kernel,
        )

        return combined_mask

    def detect_color_objects(
        self,
        hsv_image,
        class_name,
        ranges,
    ):
        detections = []

        mask = self.create_color_mask(
            hsv_image,
            ranges,
        )

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        image_area = (
            hsv_image.shape[0]
            * hsv_image.shape[1]
        )

        for contour in contours:
            area = cv2.contourArea(contour)

            if area < self.minimum_area:
                continue

            x, y, width, height = cv2.boundingRect(
                contour
            )

            detection = ObjectDetection()

            detection.class_name = class_name
            detection.confidence = float(
                min(1.0, area / (image_area * 0.03))
            )

            detection.x_min = int(x)
            detection.y_min = int(y)
            detection.x_max = int(x + width)
            detection.y_max = int(y + height)

            detection.has_position = False

            detection.position.x = 0.0
            detection.position.y = 0.0
            detection.position.z = 0.0

            detections.append(detection)

        return detections

    def image_callback(self, message):
        try:
            image = self.bridge.imgmsg_to_cv2(
                message,
                desired_encoding="bgr8",
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

        output = ObjectDetectionArray()
        output.header = message.header

        for class_name, ranges in COLOR_RANGES.items():
            detections = self.detect_color_objects(
                hsv_image,
                class_name,
                ranges,
            )

            output.detections.extend(detections)

        self.detection_publisher.publish(output)

        debug_image = image.copy()

        box_colors = {
            "red_block": (0, 0, 255),
            "green_block": (0, 255, 0),
            "blue_block": (255, 0, 0),
            "yellow_block": (0, 255, 255),
        }

        for detection in output.detections:
            color = box_colors.get(
                detection.class_name,
                (255, 255, 255),
            )

            cv2.rectangle(
                debug_image,
                (detection.x_min, detection.y_min),
                (detection.x_max, detection.y_max),
                color,
                3,
            )

            label = (
                f"{detection.class_name} "
                f"{detection.confidence:.2f}"
            )

            text_y = max(
                detection.y_min - 10,
                20,
            )

            cv2.putText(
                debug_image,
                label,
                (detection.x_min, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
            )

        debug_message = self.bridge.cv2_to_imgmsg(
            debug_image,
            encoding="bgr8",
        )

        debug_message.header = message.header

        self.debug_image_publisher.publish(
            debug_message
        )

        self.frame_count += 1

        if self.frame_count % 10 == 0:
            self.get_logger().info(
                f"当前检测到{len(output.detections)}个物块"
            )


def main(args=None):
    rclpy.init(args=args)

    node = ColorDetectorNode()

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
