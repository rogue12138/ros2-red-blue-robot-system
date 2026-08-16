#!/usr/bin/env python3

import cv2
import numpy as np
import rclpy

from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image


class MockCameraNode(Node):

    def __init__(self):
        super().__init__("mock_camera_node")

        self.publisher = self.create_publisher(
            Image,
            "/camera/image_raw",
            10,
        )

        self.bridge = CvBridge()

        self.timer = self.create_timer(
            0.2,
            self.publish_image,
        )

        self.get_logger().info(
            "模拟摄像头已启动，正在发布/camera/image_raw"
        )

    def create_test_image(self):
        image = np.full(
            (480, 640, 3),
            220,
            dtype=np.uint8,
        )

        cv2.rectangle(
            image,
            (60, 80),
            (180, 200),
            (0, 0, 255),
            -1,
        )
        cv2.putText(
            image,
            "RED",
            (85, 145),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

        cv2.rectangle(
            image,
            (240, 80),
            (360, 200),
            (0, 255, 0),
            -1,
        )
        cv2.putText(
            image,
            "GREEN",
            (252, 145),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            2,
        )

        cv2.rectangle(
            image,
            (60, 270),
            (180, 390),
            (255, 0, 0),
            -1,
        )
        cv2.putText(
            image,
            "BLUE",
            (78, 335),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

        cv2.rectangle(
            image,
            (240, 270),
            (360, 390),
            (0, 255, 255),
            -1,
        )
        cv2.putText(
            image,
            "YELLOW",
            (247, 335),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 0),
            2,
        )

        cv2.putText(
            image,
            "Mock Robot Camera",
            (410, 235),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (40, 40, 40),
            2,
        )

        return image

    def publish_image(self):
        image = self.create_test_image()

        message = self.bridge.cv2_to_imgmsg(
            image,
            encoding="bgr8",
        )

        message.header.stamp = (
            self.get_clock().now().to_msg()
        )
        message.header.frame_id = "camera_frame"

        self.publisher.publish(message)


def main(args=None):
    rclpy.init(args=args)

    node = MockCameraNode()

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
