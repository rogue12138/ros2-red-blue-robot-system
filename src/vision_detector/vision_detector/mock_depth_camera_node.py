#!/usr/bin/env python3

import numpy as np
import rclpy

from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import Image


class MockDepthCameraNode(Node):

    def __init__(self):
        super().__init__("mock_depth_camera_node")

        self.declare_parameter(
            "image_topic",
            "/camera/image_raw",
        )
        self.declare_parameter(
            "depth_topic",
            "/camera/depth/image_raw",
        )
        self.declare_parameter(
            "camera_info_topic",
            "/camera/camera_info",
        )
        self.declare_parameter(
            "camera_frame",
            "camera_frame",
        )

        self.declare_parameter(
            "depth_m",
            1.0,
        )
        self.declare_parameter(
            "depth_noise_std_m",
            0.003,
        )

        self.declare_parameter(
            "fx",
            554.256,
        )
        self.declare_parameter(
            "fy",
            554.256,
        )
        self.declare_parameter(
            "cx",
            -1.0,
        )
        self.declare_parameter(
            "cy",
            -1.0,
        )

        self.image_topic = str(
            self.get_parameter("image_topic").value
        )
        self.depth_topic = str(
            self.get_parameter("depth_topic").value
        )
        self.camera_info_topic = str(
            self.get_parameter(
                "camera_info_topic"
            ).value
        )
        self.camera_frame = str(
            self.get_parameter("camera_frame").value
        )

        self.depth_m = float(
            self.get_parameter("depth_m").value
        )
        self.depth_noise_std_m = float(
            self.get_parameter(
                "depth_noise_std_m"
            ).value
        )

        self.fx = float(
            self.get_parameter("fx").value
        )
        self.fy = float(
            self.get_parameter("fy").value
        )
        self.cx = float(
            self.get_parameter("cx").value
        )
        self.cy = float(
            self.get_parameter("cy").value
        )

        if self.depth_m <= 0.0:
            raise ValueError(
                "depth_m必须大于0"
            )

        if self.fx <= 0.0 or self.fy <= 0.0:
            raise ValueError(
                "fx和fy必须大于0"
            )

        if self.depth_noise_std_m < 0.0:
            raise ValueError(
                "depth_noise_std_m不能小于0"
            )

        self.bridge = CvBridge()
        self.random_generator = (
            np.random.default_rng(20260808)
        )

        self.depth_publisher = (
            self.create_publisher(
                Image,
                self.depth_topic,
                10,
            )
        )

        self.camera_info_publisher = (
            self.create_publisher(
                CameraInfo,
                self.camera_info_topic,
                10,
            )
        )

        self.image_subscription = (
            self.create_subscription(
                Image,
                self.image_topic,
                self.image_callback,
                10,
            )
        )

        self.first_frame_received = False

        self.get_logger().info(
            "模拟深度和相机内参节点已启动"
        )

    def create_camera_info(
        self,
        source_message,
    ):
        width = int(source_message.width)
        height = int(source_message.height)

        center_x = self.cx
        center_y = self.cy

        if center_x < 0.0:
            center_x = (
                float(width - 1) / 2.0
            )

        if center_y < 0.0:
            center_y = (
                float(height - 1) / 2.0
            )

        camera_info = CameraInfo()
        camera_info.header = source_message.header

        if not camera_info.header.frame_id:
            camera_info.header.frame_id = (
                self.camera_frame
            )

        camera_info.width = width
        camera_info.height = height
        camera_info.distortion_model = (
            "plumb_bob"
        )

        camera_info.d = [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]

        camera_info.k = [
            self.fx,
            0.0,
            center_x,
            0.0,
            self.fy,
            center_y,
            0.0,
            0.0,
            1.0,
        ]

        camera_info.r = [
            1.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            1.0,
        ]

        camera_info.p = [
            self.fx,
            0.0,
            center_x,
            0.0,
            0.0,
            self.fy,
            center_y,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
        ]

        return camera_info

    def create_depth_image(
        self,
        source_message,
    ):
        height = int(source_message.height)
        width = int(source_message.width)

        depth_image = np.full(
            (
                height,
                width,
            ),
            self.depth_m,
            dtype=np.float32,
        )

        if self.depth_noise_std_m > 0.0:
            noise = self.random_generator.normal(
                0.0,
                self.depth_noise_std_m,
                size=depth_image.shape,
            ).astype(np.float32)

            depth_image += noise

            depth_image = np.maximum(
                depth_image,
                np.float32(0.01),
            )

        depth_message = (
            self.bridge.cv2_to_imgmsg(
                depth_image,
                encoding="32FC1",
            )
        )

        depth_message.header = (
            source_message.header
        )

        if not depth_message.header.frame_id:
            depth_message.header.frame_id = (
                self.camera_frame
            )

        return depth_message

    def image_callback(self, message):
        if message.width == 0 or message.height == 0:
            self.get_logger().warning(
                "收到空图像，跳过深度发布"
            )
            return

        camera_info = self.create_camera_info(
            message
        )
        depth_message = self.create_depth_image(
            message
        )

        self.camera_info_publisher.publish(
            camera_info
        )
        self.depth_publisher.publish(
            depth_message
        )

        if not self.first_frame_received:
            self.first_frame_received = True

            self.get_logger().info(
                "已开始发布模拟深度图和相机内参："
                f"{message.width}x{message.height}，"
                f"frame_id={depth_message.header.frame_id}，"
                f"depth={self.depth_m:.3f}m"
            )


def main(args=None):
    rclpy.init(args=args)

    node = MockDepthCameraNode()

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
