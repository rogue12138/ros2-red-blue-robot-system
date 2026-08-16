#!/usr/bin/env python3

import os
import sys
import time

from pathlib import Path


YOLO_SITE_PACKAGES = os.environ.get(
    "YOLO_SITE_PACKAGES",
    str(
        Path.home()
        / "robot_ws"
        / "venvs"
        / "yolo_cpu"
        / "lib"
        / "python3.10"
        / "site-packages"
    ),
)

if (
    Path(YOLO_SITE_PACKAGES).is_dir()
    and YOLO_SITE_PACKAGES not in sys.path
):
    sys.path.insert(0, YOLO_SITE_PACKAGES)


import cv2
import rclpy

from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import HistoryPolicy
from rclpy.qos import QoSProfile
from rclpy.qos import ReliabilityPolicy
from sensor_msgs.msg import Image
from ultralytics import YOLO

from robot_interfaces.msg import ObjectDetection
from robot_interfaces.msg import ObjectDetectionArray


ALLOWED_CLASSES = {
    "red_cube",
    "blue_cube",
}

BOX_COLORS = {
    "red_cube": (0, 0, 255),
    "blue_cube": (255, 0, 0),
}


class YoloDetectorNode(Node):

    def __init__(self):
        super().__init__("yolo_detector_node")

        default_model_path = str(
            Path.home()
            / "robot_ws"
            / "models"
            / "final"
            / "red_blue_yolo11n_manual180_960_best.pt"
        )

        self.declare_parameter(
            "model_path",
            default_model_path,
        )
        self.declare_parameter(
            "input_topic",
            "/camera/image_raw",
        )
        self.declare_parameter(
            "detections_topic",
            "/vision/yolo_detections",
        )
        self.declare_parameter(
            "debug_image_topic",
            "/vision/yolo_debug_image",
        )
        self.declare_parameter(
            "confidence_threshold",
            0.05,
        )
        self.declare_parameter(
            "nms_iou_threshold",
            0.50,
        )
        self.declare_parameter(
            "image_size",
            960,
        )
        self.declare_parameter(
            "maximum_detections",
            100,
        )
        self.declare_parameter(
            "device",
            "cpu",
        )
        self.declare_parameter(
            "log_every_n_frames",
            20,
        )

        self.model_path = os.path.expanduser(
            str(
                self.get_parameter(
                    "model_path"
                ).value
            )
        )
        self.input_topic = str(
            self.get_parameter(
                "input_topic"
            ).value
        )
        self.detections_topic = str(
            self.get_parameter(
                "detections_topic"
            ).value
        )
        self.debug_image_topic = str(
            self.get_parameter(
                "debug_image_topic"
            ).value
        )
        self.confidence_threshold = float(
            self.get_parameter(
                "confidence_threshold"
            ).value
        )
        self.nms_iou_threshold = float(
            self.get_parameter(
                "nms_iou_threshold"
            ).value
        )
        self.image_size = int(
            self.get_parameter(
                "image_size"
            ).value
        )
        self.maximum_detections = int(
            self.get_parameter(
                "maximum_detections"
            ).value
        )
        self.device = str(
            self.get_parameter(
                "device"
            ).value
        )
        self.log_every_n_frames = int(
            self.get_parameter(
                "log_every_n_frames"
            ).value
        )

        if not Path(self.model_path).is_file():
            raise FileNotFoundError(
                f"YOLO模型不存在：{self.model_path}"
            )

        if not 0.0 < self.confidence_threshold <= 1.0:
            raise ValueError(
                "confidence_threshold必须在0到1之间"
            )

        if not 0.0 < self.nms_iou_threshold <= 1.0:
            raise ValueError(
                "nms_iou_threshold必须在0到1之间"
            )

        if self.image_size < 32:
            raise ValueError(
                "image_size不能小于32"
            )

        if self.maximum_detections < 1:
            raise ValueError(
                "maximum_detections必须大于0"
            )

        if self.log_every_n_frames < 1:
            self.log_every_n_frames = 1

        self.bridge = CvBridge()
        self.frame_count = 0
        self.last_inference_ms = 0.0

        self.get_logger().info(
            f"正在加载YOLO模型：{self.model_path}"
        )

        self.model = YOLO(self.model_path)

        model_classes = {
            str(class_name)
            for class_name in self.model.names.values()
        }

        missing_classes = (
            ALLOWED_CLASSES - model_classes
        )

        if missing_classes:
            raise ValueError(
                "模型缺少类别："
                f"{sorted(missing_classes)}"
            )

        sensor_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )

        self.image_subscription = (
            self.create_subscription(
                Image,
                self.input_topic,
                self.image_callback,
                sensor_qos,
            )
        )

        self.detection_publisher = (
            self.create_publisher(
                ObjectDetectionArray,
                self.detections_topic,
                10,
            )
        )

        self.debug_image_publisher = (
            self.create_publisher(
                Image,
                self.debug_image_topic,
                10,
            )
        )

        self.get_logger().info(
            "YOLO检测节点已启动："
            f"input={self.input_topic}，"
            f"output={self.detections_topic}，"
            f"imgsz={self.image_size}，"
            f"conf={self.confidence_threshold}，"
            f"device={self.device}"
        )
    def refine_bbox_with_hsv(
        self,
        image,
        class_name,
        coordinates,
    ):
        image_height, image_width = image.shape[:2]

        original_x_min = int(round(coordinates[0]))
        original_y_min = int(round(coordinates[1]))
        original_x_max = int(round(coordinates[2]))
        original_y_max = int(round(coordinates[3]))

        original_x_min = max(
            0,
            min(image_width - 1, original_x_min),
        )
        original_y_min = max(
            0,
            min(image_height - 1, original_y_min),
        )
        original_x_max = max(
            original_x_min + 1,
            min(image_width, original_x_max),
        )
        original_y_max = max(
            original_y_min + 1,
            min(image_height, original_y_max),
        )

        original_width = (
            original_x_max - original_x_min
        )
        original_height = (
            original_y_max - original_y_min
        )
        original_area = float(
            original_width * original_height
        )

        if original_area <= 0.0:
            return coordinates

        expansion = max(
            6,
            int(round(
                max(
                    original_width,
                    original_height,
                )
                * 0.20
            )),
        )

        roi_x_min = max(
            0,
            original_x_min - expansion,
        )
        roi_y_min = max(
            0,
            original_y_min - expansion,
        )
        roi_x_max = min(
            image_width,
            original_x_max + expansion,
        )
        roi_y_max = min(
            image_height,
            original_y_max + expansion,
        )

        roi = image[
            roi_y_min:roi_y_max,
            roi_x_min:roi_x_max,
        ]

        if roi.size == 0:
            return coordinates

        hsv_roi = cv2.cvtColor(
            roi,
            cv2.COLOR_BGR2HSV,
        )

        if class_name == "red_cube":
            first_mask = cv2.inRange(
                hsv_roi,
                (0, 80, 50),
                (12, 255, 255),
            )
            second_mask = cv2.inRange(
                hsv_roi,
                (168, 80, 50),
                (179, 255, 255),
            )
            mask = cv2.bitwise_or(
                first_mask,
                second_mask,
            )
        elif class_name == "blue_cube":
            mask = cv2.inRange(
                hsv_roi,
                (95, 70, 40),
                (135, 255, 255),
            )
        else:
            return coordinates

        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (3, 3),
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

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        best_coordinates = None
        best_score = None

        original_center_x = (
            original_x_min + original_x_max
        ) / 2.0
        original_center_y = (
            original_y_min + original_y_max
        ) / 2.0

        minimum_contour_area = max(
            30.0,
            original_area * 0.03,
        )

        for contour in contours:
            contour_area = float(
                cv2.contourArea(contour)
            )

            if contour_area < minimum_contour_area:
                continue

            x, y, width, height = cv2.boundingRect(
                contour
            )

            if width <= 0 or height <= 0:
                continue

            candidate_x_min = roi_x_min + x
            candidate_y_min = roi_y_min + y
            candidate_x_max = (
                candidate_x_min + width
            )
            candidate_y_max = (
                candidate_y_min + height
            )

            if (
                width > original_width * 1.80
                or height > original_height * 1.80
            ):
                continue

            intersection_x_min = max(
                original_x_min,
                candidate_x_min,
            )
            intersection_y_min = max(
                original_y_min,
                candidate_y_min,
            )
            intersection_x_max = min(
                original_x_max,
                candidate_x_max,
            )
            intersection_y_max = min(
                original_y_max,
                candidate_y_max,
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
            intersection_area = float(
                intersection_width
                * intersection_height
            )

            if intersection_area <= 0.0:
                continue

            candidate_area = float(
                width * height
            )
            union_area = (
                original_area
                + candidate_area
                - intersection_area
            )

            if union_area <= 0.0:
                continue

            iou = intersection_area / union_area

            if iou < 0.05:
                continue

            candidate_center_x = (
                candidate_x_min
                + candidate_x_max
            ) / 2.0
            candidate_center_y = (
                candidate_y_min
                + candidate_y_max
            ) / 2.0

            center_distance = (
                (
                    candidate_center_x
                    - original_center_x
                )
                ** 2
                + (
                    candidate_center_y
                    - original_center_y
                )
                ** 2
            ) ** 0.5

            normalized_distance = (
                center_distance
                / max(
                    1.0,
                    float(max(
                        original_width,
                        original_height,
                    )),
                )
            )

            area_similarity = (
                min(
                    original_area,
                    candidate_area,
                )
                / max(
                    original_area,
                    candidate_area,
                )
            )

            score = (
                2.0 * iou
                + 0.5 * area_similarity
                - 0.5 * normalized_distance
            )

            if (
                best_score is None
                or score > best_score
            ):
                best_score = score

                if class_name == "red_cube":
                    padding_pixels = 10
                else:
                    padding_pixels = 10

                refined_x_min = max(
                    0,
                    candidate_x_min - padding_pixels,
                )
                refined_y_min = max(
                    0,
                    candidate_y_min - padding_pixels,
                )
                refined_x_max = min(
                    image_width,
                    candidate_x_max + padding_pixels,
                )
                refined_y_max = min(
                    image_height,
                    candidate_y_max + padding_pixels,
                )

                best_coordinates = [
                    float(refined_x_min),
                    float(refined_y_min),
                    float(refined_x_max),
                    float(refined_y_max),
                ]

        if best_coordinates is None:
            return coordinates

        return best_coordinates
    def create_detection(
        self,
        class_name,
        confidence,
        coordinates,
        detection_id,
    ):
        x_min, y_min, x_max, y_max = coordinates

        detection = ObjectDetection()

        detection.class_name = class_name
        detection.confidence = float(confidence)

        detection.x_min = int(round(x_min))
        detection.y_min = int(round(y_min))
        detection.x_max = int(round(x_max))
        detection.y_max = int(round(y_max))

        detection.has_position = False
        detection.position.x = 0.0
        detection.position.y = 0.0
        detection.position.z = 0.0

        if hasattr(detection, "detection_id"):
            detection.detection_id = int(
                detection_id
            )

        center_x = int(
            round((x_min + x_max) / 2.0)
        )
        center_y = int(
            round((y_min + y_max) / 2.0)
        )

        if hasattr(detection, "center_x"):
            detection.center_x = center_x

        if hasattr(detection, "center_y"):
            detection.center_y = center_y

        return detection

    def draw_detection(
        self,
        image,
        detection,
        detection_id,
    ):
        color = BOX_COLORS[
            detection.class_name
        ]

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

        cv2.drawMarker(
            image,
            (center_x, center_y),
            color,
            markerType=cv2.MARKER_CROSS,
            markerSize=12,
            thickness=2,
        )

        label = (
            f"YOLO ID:{detection_id} "
            f"{detection.class_name} "
            f"{detection.confidence:.2f}"
        )

        text_y = max(
            22,
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
            0.52,
            color,
            2,
            cv2.LINE_AA,
        )

    def run_inference(self, image):
        start_time = time.perf_counter()

        results = self.model.predict(
            source=image,
            imgsz=self.image_size,
            conf=self.confidence_threshold,
            iou=self.nms_iou_threshold,
            max_det=self.maximum_detections,
            device=self.device,
            verbose=False,
        )

        self.last_inference_ms = (
            time.perf_counter() - start_time
        ) * 1000.0

        if not results:
            return []

        result = results[0]

        if result.boxes is None:
            return []

        xyxy_values = (
            result.boxes.xyxy
            .detach()
            .cpu()
            .tolist()
        )
        confidence_values = (
            result.boxes.conf
            .detach()
            .cpu()
            .tolist()
        )
        class_values = (
            result.boxes.cls
            .detach()
            .cpu()
            .tolist()
        )

        detections = []

        for coordinates, confidence, class_id in zip(
            xyxy_values,
            confidence_values,
            class_values,
        ):
            class_id = int(class_id)

            class_name = str(
                self.model.names.get(
                    class_id,
                    "unknown",
                )
            )

            if class_name not in ALLOWED_CLASSES:
                continue

            detections.append(
                (
                    class_name,
                    float(confidence),
                    coordinates,
                )
            )

        detections.sort(
            key=lambda item: (
                item[2][0],
                item[2][1],
                item[0],
            )
        )

        return detections

    def image_callback(self, message):
        try:
            image = self.bridge.imgmsg_to_cv2(
                message,
                desired_encoding="bgr8",
            )
        except Exception as error:
            self.get_logger().error(
                f"输入图像转换失败：{error}"
            )
            return

        try:
            raw_detections = self.run_inference(
                image
            )
        except Exception as error:
            self.get_logger().error(
                f"YOLO推理失败：{error}"
            )
            return

        detection_message = (
            ObjectDetectionArray()
        )
        detection_message.header = (
            message.header
        )

        debug_image = image.copy()
        output_detections = []

        for detection_id, raw_detection in enumerate(
            raw_detections,
            start=1,
        ):
            (
                class_name,
                confidence,
                coordinates,
            ) = raw_detection

            refined_coordinates = (
                self.refine_bbox_with_hsv(
                    image,
                    class_name,
                    coordinates,
                )
            )

            detection = self.create_detection(
                class_name,
                confidence,
                refined_coordinates,
                detection_id,
            )

            output_detections.append(detection)

            self.draw_detection(
                debug_image,
                detection,
                detection_id,
            )

        detection_message.detections = (
            output_detections
        )

        self.detection_publisher.publish(
            detection_message
        )

        try:
            debug_message = (
                self.bridge.cv2_to_imgmsg(
                    debug_image,
                    encoding="bgr8",
                )
            )
            debug_message.header = message.header

            self.debug_image_publisher.publish(
                debug_message
            )
        except Exception as error:
            self.get_logger().error(
                f"YOLO调试图像发布失败：{error}"
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
                for detection
                in output_detections
            )
            blue_count = sum(
                detection.class_name
                == "blue_cube"
                for detection
                in output_detections
            )

            self.get_logger().info(
                "YOLO检测结果："
                f"red_cube={red_count}，"
                f"blue_cube={blue_count}，"
                f"inference={self.last_inference_ms:.1f}ms"
            )


def main(args=None):
    rclpy.init(args=args)

    node = None

    try:
        node = YoloDetectorNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as error:
        if node is not None:
            node.get_logger().fatal(
                f"YOLO节点异常：{error}"
            )
        else:
            print(
                f"YOLO节点启动失败：{error}",
                file=sys.stderr,
            )
        raise
    finally:
        if node is not None:
            node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
