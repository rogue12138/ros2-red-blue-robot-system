#!/usr/bin/env python3

import json
from datetime import datetime
from pathlib import Path

import cv2
import rclpy

from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String

from robot_interfaces.msg import ObjectDetectionArray


CLASS_IDS = {
    "red_cube": 0,
    "blue_cube": 1,
}


class DatasetCaptureNode(Node):

    def __init__(self):
        super().__init__(
            "dataset_capture_node"
        )

        self.declare_parameter(
            "output_directory",
            "~/robot_ws/datasets/"
            "color_cubes_preliminary",
        )
        self.declare_parameter(
            "run_name",
            "",
        )
        self.declare_parameter(
            "max_images",
            200,
        )
        self.declare_parameter(
            "jpeg_quality",
            95,
        )

        output_directory = str(
            self.get_parameter(
                "output_directory"
            ).value
        )
        run_name = str(
            self.get_parameter(
                "run_name"
            ).value
        ).strip()

        self.max_images = int(
            self.get_parameter(
                "max_images"
            ).value
        )
        self.jpeg_quality = int(
            self.get_parameter(
                "jpeg_quality"
            ).value
        )

        self.jpeg_quality = min(
            100,
            max(1, self.jpeg_quality),
        )

        if not run_name:
            run_name = datetime.now().strftime(
                "capture_%Y%m%d_%H%M%S"
            )

        self.dataset_directory = (
            Path(output_directory)
            .expanduser()
            .resolve()
            / run_name
        )

        self.image_directory = (
            self.dataset_directory
            / "images"
            / "all"
        )
        self.label_directory = (
            self.dataset_directory
            / "labels"
            / "all"
        )
        self.metadata_directory = (
            self.dataset_directory
            / "metadata"
        )

        self.image_directory.mkdir(
            parents=True,
            exist_ok=True,
        )
        self.label_directory.mkdir(
            parents=True,
            exist_ok=True,
        )
        self.metadata_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.metadata_file = (
            self.metadata_directory
            / "scenes.jsonl"
        )

        self.bridge = CvBridge()

        self.latest_image = None
        self.latest_image_stamp = None

        self.latest_ground_truth = None
        self.latest_ground_truth_stamp = None

        self.pending_scene_info = None
        self.saved_scene_indices = set()

        self.saved_count = 0
        self.capture_complete = False

        self.image_subscription = (
            self.create_subscription(
                Image,
                "/camera/image_raw",
                self.image_callback,
                10,
            )
        )

        self.ground_truth_subscription = (
            self.create_subscription(
                ObjectDetectionArray,
                "/simulation/ground_truth_detections",
                self.ground_truth_callback,
                10,
            )
        )

        self.scene_info_subscription = (
            self.create_subscription(
                String,
                "/simulation/scene_info",
                self.scene_info_callback,
                10,
            )
        )

        self.status_publisher = (
            self.create_publisher(
                String,
                "/dataset_capture/status",
                10,
            )
        )

        self.get_logger().info(
            "数据集采集节点已启动"
        )
        self.get_logger().info(
            "数据集目录："
            f"{self.dataset_directory}"
        )
        self.get_logger().info(
            f"最大采集数量：{self.max_images}"
        )

    def stamp_key(self, header):
        return (
            int(header.stamp.sec),
            int(header.stamp.nanosec),
        )

    def image_callback(self, message):
        try:
            self.latest_image = (
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

        self.latest_image_stamp = (
            self.stamp_key(message.header)
        )

        self.try_capture()

    def ground_truth_callback(
        self,
        message,
    ):
        self.latest_ground_truth = message
        self.latest_ground_truth_stamp = (
            self.stamp_key(message.header)
        )

        self.try_capture()

    def scene_info_callback(self, message):
        try:
            scene_info = json.loads(
                message.data
            )
        except json.JSONDecodeError as error:
            self.get_logger().error(
                f"场景信息不是合法JSON：{error}"
            )
            return

        if "scene_index" not in scene_info:
            self.get_logger().error(
                "场景信息缺少scene_index"
            )
            return

        scene_index = int(
            scene_info["scene_index"]
        )

        if (
            scene_index
            in self.saved_scene_indices
        ):
            return

        self.pending_scene_info = scene_info

        self.try_capture()

    def try_capture(self):
        if self.capture_complete:
            return

        if self.pending_scene_info is None:
            return

        if self.latest_image is None:
            return

        if self.latest_ground_truth is None:
            return

        if (
            self.latest_image_stamp
            != self.latest_ground_truth_stamp
        ):
            return

        scene_index = int(
            self.pending_scene_info[
                "scene_index"
            ]
        )

        if (
            scene_index
            in self.saved_scene_indices
        ):
            return

        self.save_sample(
            scene_index,
            self.pending_scene_info,
            self.latest_image,
            self.latest_ground_truth,
        )

    def clip_value(
        self,
        value,
        minimum,
        maximum,
    ):
        return min(
            maximum,
            max(minimum, value),
        )

    def make_yolo_lines(
        self,
        detections,
        image_width,
        image_height,
    ):
        label_lines = []

        for detection in detections:
            class_name = (
                detection.class_name
            )

            if class_name not in CLASS_IDS:
                continue

            x_min = self.clip_value(
                float(detection.x_min),
                0.0,
                float(image_width),
            )
            y_min = self.clip_value(
                float(detection.y_min),
                0.0,
                float(image_height),
            )
            x_max = self.clip_value(
                float(detection.x_max),
                0.0,
                float(image_width),
            )
            y_max = self.clip_value(
                float(detection.y_max),
                0.0,
                float(image_height),
            )

            box_width = x_max - x_min
            box_height = y_max - y_min

            if (
                box_width <= 1.0
                or box_height <= 1.0
            ):
                continue

            center_x = (
                x_min + x_max
            ) / 2.0
            center_y = (
                y_min + y_max
            ) / 2.0

            normalized_center_x = (
                center_x / image_width
            )
            normalized_center_y = (
                center_y / image_height
            )
            normalized_width = (
                box_width / image_width
            )
            normalized_height = (
                box_height / image_height
            )

            class_id = CLASS_IDS[
                class_name
            ]

            label_lines.append(
                f"{class_id} "
                f"{normalized_center_x:.6f} "
                f"{normalized_center_y:.6f} "
                f"{normalized_width:.6f} "
                f"{normalized_height:.6f}"
            )

        return label_lines

    def save_sample(
        self,
        scene_index,
        scene_info,
        image,
        ground_truth,
    ):
        file_stem = (
            f"scene_{scene_index:06d}"
        )

        image_path = (
            self.image_directory
            / f"{file_stem}.jpg"
        )
        label_path = (
            self.label_directory
            / f"{file_stem}.txt"
        )

        image_saved = cv2.imwrite(
            str(image_path),
            image,
            [
                cv2.IMWRITE_JPEG_QUALITY,
                self.jpeg_quality,
            ],
        )

        if not image_saved:
            self.get_logger().error(
                f"图片保存失败：{image_path}"
            )
            return

        image_height, image_width = (
            image.shape[:2]
        )

        label_lines = self.make_yolo_lines(
            ground_truth.detections,
            image_width,
            image_height,
        )

        label_text = "\n".join(
            label_lines
        )

        if label_text:
            label_text += "\n"

        label_path.write_text(
            label_text,
            encoding="utf-8",
        )

        metadata = dict(scene_info)
        metadata.update(
            {
                "image_file": str(
                    image_path.relative_to(
                        self.dataset_directory
                    )
                ),
                "label_file": str(
                    label_path.relative_to(
                        self.dataset_directory
                    )
                ),
                "label_count": len(
                    label_lines
                ),
                "image_width": image_width,
                "image_height": image_height,
            }
        )

        with self.metadata_file.open(
            "a",
            encoding="utf-8",
        ) as metadata_stream:
            metadata_stream.write(
                json.dumps(
                    metadata,
                    ensure_ascii=False,
                )
                + "\n"
            )

        self.saved_scene_indices.add(
            scene_index
        )
        self.saved_count += 1
        self.pending_scene_info = None

        status = {
            "saved_count": self.saved_count,
            "max_images": self.max_images,
            "scene_index": scene_index,
            "dataset_directory": str(
                self.dataset_directory
            ),
        }

        status_message = String()
        status_message.data = json.dumps(
            status,
            ensure_ascii=False,
        )
        self.status_publisher.publish(
            status_message
        )

        self.get_logger().info(
            "已保存数据："
            f"{self.saved_count}/"
            f"{self.max_images}，"
            f"scene={scene_index}，"
            f"labels={len(label_lines)}"
        )

        if (
            self.max_images > 0
            and self.saved_count
            >= self.max_images
        ):
            self.capture_complete = True

            self.get_logger().info(
                "数据集采集完成："
                f"{self.dataset_directory}"
            )

            complete_message = String()
            complete_message.data = (
                json.dumps(
                    {
                        "complete": True,
                        "saved_count": (
                            self.saved_count
                        ),
                        "dataset_directory": str(
                            self.dataset_directory
                        ),
                    },
                    ensure_ascii=False,
                )
            )
            self.status_publisher.publish(
                complete_message
            )


def main(args=None):
    rclpy.init(args=args)

    node = DatasetCaptureNode()

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
