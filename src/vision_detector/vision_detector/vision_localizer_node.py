#!/usr/bin/env python3

import numpy as np
import rclpy
import math

from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer
from message_filters import Subscriber
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import Image
from rclpy.duration import Duration
from rclpy.time import Time
from tf2_ros import Buffer
from tf2_ros import TransformException
from tf2_ros import TransformListener
from collections import deque

from robot_interfaces.msg import ObjectDetection
from robot_interfaces.msg import ObjectDetectionArray


class VisionLocalizerNode(Node):

    def __init__(self):
        super().__init__("vision_localizer_node")

        self.declare_parameter(
            "detections_topic",
            "/vision/detections",
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
            "output_topic",
            "/vision/localized_detections",
        )
        self.declare_parameter(
            "target_frame",
            "base_link",
        )
        self.declare_parameter(
            "tf_timeout_sec",
            0.20,
        )
        self.declare_parameter(
            "depth_window_size",
            5,
        )
        self.declare_parameter(
            "minimum_depth_m",
            0.10,
        )
        self.declare_parameter(
            "maximum_depth_m",
            5.00,
        )
        self.declare_parameter(
            "integer_depth_scale",
            0.001,
        )

        self.declare_parameter(
            "synchronization_queue_size",
            20,
        )
        self.declare_parameter(
            "synchronization_slop_sec",
            0.05,
        )
        self.declare_parameter(
            "log_every_n_frames",
            30,
        )
        self.declare_parameter(
            "position_filter_window",
            5,
        )
        self.declare_parameter(
            "track_match_distance_pixels",
            80.0,
        )
        self.declare_parameter(
            "track_timeout_frames",
            10,
        )
        self.detections_topic = str(
            self.get_parameter(
                "detections_topic"
            ).value
        )
        self.depth_topic = str(
            self.get_parameter(
                "depth_topic"
            ).value
        )
        self.camera_info_topic = str(
            self.get_parameter(
                "camera_info_topic"
            ).value
        )
        self.output_topic = str(
            self.get_parameter(
                "output_topic"
            ).value
        )
        self.target_frame = str(
            self.get_parameter(
                "target_frame"
            ).value
        )
        self.tf_timeout_sec = float(
            self.get_parameter(
                "tf_timeout_sec"
            ).value
        )
        self.depth_window_size = int(
            self.get_parameter(
                "depth_window_size"
            ).value
        )
        self.minimum_depth_m = float(
            self.get_parameter(
                "minimum_depth_m"
            ).value
        )
        self.maximum_depth_m = float(
            self.get_parameter(
                "maximum_depth_m"
            ).value
        )
        self.integer_depth_scale = float(
            self.get_parameter(
                "integer_depth_scale"
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
        self.log_every_n_frames = int(
            self.get_parameter(
                "log_every_n_frames"
            ).value
        )
        self.position_filter_window = int(
            self.get_parameter(
                "position_filter_window"
            ).value
        )
        self.track_match_distance_pixels = float(
            self.get_parameter(
                "track_match_distance_pixels"
            ).value
        )
        self.track_timeout_frames = int(
            self.get_parameter(
                "track_timeout_frames"
            ).value
        )
        if not self.target_frame:
            raise ValueError(
                "target_frame不能为空"
            )

        if self.tf_timeout_sec <= 0.0:
            raise ValueError(
                "tf_timeout_sec必须大于0"
            )

        if self.depth_window_size < 1:
            self.depth_window_size = 1

        if self.depth_window_size % 2 == 0:
            self.depth_window_size += 1

        if self.minimum_depth_m <= 0.0:
            raise ValueError(
                "minimum_depth_m必须大于0"
            )

        if (
            self.maximum_depth_m
            <= self.minimum_depth_m
        ):
            raise ValueError(
                "maximum_depth_m必须大于minimum_depth_m"
            )

        if self.integer_depth_scale <= 0.0:
            raise ValueError(
                "integer_depth_scale必须大于0"
            )

        if queue_size < 1:
            queue_size = 1

        if synchronization_slop < 0.0:
            synchronization_slop = 0.0

        if self.log_every_n_frames < 1:
            self.log_every_n_frames = 1
        if self.position_filter_window < 1:
            raise ValueError(
                "position_filter_window必须大于0"
            )

        if self.track_match_distance_pixels <= 0.0:
            raise ValueError(
                "track_match_distance_pixels必须大于0"
            )

        if self.track_timeout_frames < 1:
            raise ValueError(
                "track_timeout_frames必须大于0"
            )
        self.bridge = CvBridge()
        self.frame_count = 0
        self.tf_buffer = Buffer()

        self.tf_listener = TransformListener(
            self.tf_buffer,
            self,
        )

        self.tf_failure_count = 0
        self.tracks = {}
        self.next_track_id = 1
        self.localized_publisher = (
            self.create_publisher(
                ObjectDetectionArray,
                self.output_topic,
                10,
            )
        )

        self.detections_subscriber = Subscriber(
            self,
            ObjectDetectionArray,
            self.detections_topic,
            qos_profile=10,
        )

        self.depth_subscriber = Subscriber(
            self,
            Image,
            self.depth_topic,
            qos_profile=10,
        )

        self.camera_info_subscriber = Subscriber(
            self,
            CameraInfo,
            self.camera_info_topic,
            qos_profile=10,
        )

        self.synchronizer = (
            ApproximateTimeSynchronizer(
                [
                    self.detections_subscriber,
                    self.depth_subscriber,
                    self.camera_info_subscriber,
                ],
                queue_size=queue_size,
                slop=synchronization_slop,
            )
        )

        self.synchronizer.registerCallback(
            self.synchronized_callback
        )

        self.get_logger().info(
            "深度反投影视觉定位节点已启动"
        )

    def copy_detection(self, source):
        target = ObjectDetection()

        target.class_name = source.class_name
        target.confidence = source.confidence

        target.x_min = source.x_min
        target.y_min = source.y_min
        target.x_max = source.x_max
        target.y_max = source.y_max

        target.has_position = False
        target.position.x = 0.0
        target.position.y = 0.0
        target.position.z = 0.0

        return target

    def obtain_depth_m(
        self,
        depth_image,
        depth_encoding,
        center_x,
        center_y,
    ):
        image_height, image_width = (
            depth_image.shape[:2]
        )

        if (
            center_x < 0
            or center_x >= image_width
            or center_y < 0
            or center_y >= image_height
        ):
            return None

        half_window = (
            self.depth_window_size // 2
        )

        x_start = max(
            0,
            center_x - half_window,
        )
        x_end = min(
            image_width,
            center_x + half_window + 1,
        )
        y_start = max(
            0,
            center_y - half_window,
        )
        y_end = min(
            image_height,
            center_y + half_window + 1,
        )

        depth_region = np.asarray(
            depth_image[
                y_start:y_end,
                x_start:x_end,
            ],
            dtype=np.float64,
        )

        encoding = depth_encoding.upper()

        if encoding in {
            "16UC1",
            "MONO16",
        }:
            depth_region *= (
                self.integer_depth_scale
            )

        valid_values = depth_region[
            np.isfinite(depth_region)
        ]

        valid_values = valid_values[
            (
                valid_values
                >= self.minimum_depth_m
            )
            & (
                valid_values
                <= self.maximum_depth_m
            )
        ]

        if valid_values.size == 0:
            return None

        return float(
            np.median(valid_values)
        )

    def back_project(
        self,
        center_x,
        center_y,
        depth_m,
        camera_info,
    ):
        fx = float(camera_info.k[0])
        fy = float(camera_info.k[4])
        cx = float(camera_info.k[2])
        cy = float(camera_info.k[5])

        if fx <= 0.0 or fy <= 0.0:
            return None

        position_x = (
            (float(center_x) - cx)
            * depth_m
            / fx
        )
        position_y = (
            (float(center_y) - cy)
            * depth_m
            / fy
        )
        position_z = depth_m

        return (
            position_x,
            position_y,
            position_z,
        )
    def rotate_position(
        self,
        position,
        rotation,
    ):
        quaternion_norm = math.sqrt(
            rotation.x * rotation.x
            + rotation.y * rotation.y
            + rotation.z * rotation.z
            + rotation.w * rotation.w
        )

        if quaternion_norm <= 1.0e-12:
            return None

        qx = rotation.x / quaternion_norm
        qy = rotation.y / quaternion_norm
        qz = rotation.z / quaternion_norm
        qw = rotation.w / quaternion_norm

        position_x = float(position[0])
        position_y = float(position[1])
        position_z = float(position[2])

        temporary_x = 2.0 * (
            qy * position_z
            - qz * position_y
        )
        temporary_y = 2.0 * (
            qz * position_x
            - qx * position_z
        )
        temporary_z = 2.0 * (
            qx * position_y
            - qy * position_x
        )

        rotated_x = (
            position_x
            + qw * temporary_x
            + qy * temporary_z
            - qz * temporary_y
        )
        rotated_y = (
            position_y
            + qw * temporary_y
            + qz * temporary_x
            - qx * temporary_z
        )
        rotated_z = (
            position_z
            + qw * temporary_z
            + qx * temporary_y
            - qy * temporary_x
        )

        return (
            rotated_x,
            rotated_y,
            rotated_z,
        )

    def transform_position(
        self,
        position,
        source_frame,
        source_stamp,
    ):
        if source_frame == self.target_frame:
            return position

        if not source_frame:
            return None

        try:
            transform_stamped = (
                self.tf_buffer.lookup_transform(
                    self.target_frame,
                    source_frame,
                    Time.from_msg(source_stamp),
                    timeout=Duration(
                        seconds=self.tf_timeout_sec
                    ),
                )
            )
        except TransformException as error:
            self.tf_failure_count += 1

            if (
                self.tf_failure_count == 1
                or self.tf_failure_count
                % self.log_every_n_frames
                == 0
            ):
                self.get_logger().warning(
                    "TF转换失败："
                    f"{source_frame}→"
                    f"{self.target_frame}，"
                    f"{error}"
                )

            return None

        rotated_position = self.rotate_position(
            position,
            transform_stamped.transform.rotation,
        )

        if rotated_position is None:
            return None

        translation = (
            transform_stamped.transform.translation
        )

        self.tf_failure_count = 0

        return (
            rotated_position[0] + translation.x,
            rotated_position[1] + translation.y,
            rotated_position[2] + translation.z,
        )
    def create_track(
        self,
        class_name,
        center_x,
        center_y,
    ):
        track_id = self.next_track_id
        self.next_track_id += 1

        self.tracks[track_id] = {
            "class_name": class_name,
            "center_x": float(center_x),
            "center_y": float(center_y),
            "positions": deque(
                maxlen=self.position_filter_window
            ),
            "last_seen_frame": self.frame_count,
        }

        return track_id

    def match_track(
        self,
        class_name,
        center_x,
        center_y,
        used_track_ids,
    ):
        best_track_id = None
        best_distance = None

        for track_id, track in self.tracks.items():
            if track_id in used_track_ids:
                continue

            if track["class_name"] != class_name:
                continue

            distance = math.hypot(
                float(center_x)
                - track["center_x"],
                float(center_y)
                - track["center_y"],
            )

            if (
                distance
                > self.track_match_distance_pixels
            ):
                continue

            if (
                best_distance is None
                or distance < best_distance
            ):
                best_track_id = track_id
                best_distance = distance

        if best_track_id is None:
            best_track_id = self.create_track(
                class_name,
                center_x,
                center_y,
            )

        used_track_ids.add(best_track_id)

        return best_track_id

    def update_track(
        self,
        track_id,
        center_x,
        center_y,
        position,
    ):
        track = self.tracks[track_id]

        track["center_x"] = float(center_x)
        track["center_y"] = float(center_y)
        track["last_seen_frame"] = (
            self.frame_count
        )

        if position is None:
            track["positions"].clear()
            return None

        track["positions"].append(
            (
                float(position[0]),
                float(position[1]),
                float(position[2]),
            )
        )

        if (
            len(track["positions"])
            < self.position_filter_window
        ):
            return None

        position_array = np.asarray(
            list(track["positions"]),
            dtype=np.float64,
        )

        median_position = np.median(
            position_array,
            axis=0,
        )

        return (
            float(median_position[0]),
            float(median_position[1]),
            float(median_position[2]),
        )

    def remove_expired_tracks(self):
        expired_track_ids = []

        for track_id, track in self.tracks.items():
            missing_frames = (
                self.frame_count
                - track["last_seen_frame"]
            )

            if (
                missing_frames
                > self.track_timeout_frames
            ):
                expired_track_ids.append(
                    track_id
                )

        for track_id in expired_track_ids:
            del self.tracks[track_id]
    def synchronized_callback(
        self,
        detections_message,
        depth_message,
        camera_info_message,
    ):
        try:
            depth_image = (
                self.bridge.imgmsg_to_cv2(
                    depth_message,
                    desired_encoding="passthrough",
                )
            )
        except Exception as error:
            self.get_logger().error(
                f"深度图转换失败：{error}"
            )
            return

        output_message = ObjectDetectionArray()
        output_message.header = (
            detections_message.header
        )
        output_message.header.frame_id = (
            self.target_frame
        )

        camera_frame = (
            camera_info_message.header.frame_id
            or depth_message.header.frame_id
            or detections_message.header.frame_id
        )

        valid_position_count = 0
        used_track_ids = set()

        for source_detection in (
            detections_message.detections
        ):
            target_detection = (
                self.copy_detection(
                    source_detection
                )
            )

            center_x = int(
                (
                    source_detection.x_min
                    + source_detection.x_max
                )
                / 2
            )
            center_y = int(
                (
                    source_detection.y_min
                    + source_detection.y_max
                )
                / 2
            )

            track_id = self.match_track(
                source_detection.class_name,
                center_x,
                center_y,
                used_track_ids,
            )

            target_position = None

            depth_m = self.obtain_depth_m(
                depth_image,
                depth_message.encoding,
                center_x,
                center_y,
            )

            if depth_m is not None:
                camera_position = self.back_project(
                    center_x,
                    center_y,
                    depth_m,
                    camera_info_message,
                )

                if camera_position is not None:
                    target_position = (
                        self.transform_position(
                            camera_position,
                            camera_frame,
                            detections_message.header.stamp,
                        )
                    )

            filtered_position = self.update_track(
                track_id,
                center_x,
                center_y,
                target_position,
            )

            if filtered_position is not None:
                target_detection.has_position = (
                    True
                )
                target_detection.position.x = (
                    filtered_position[0]
                )
                target_detection.position.y = (
                    filtered_position[1]
                )
                target_detection.position.z = (
                    filtered_position[2]
                )

                valid_position_count += 1

            output_message.detections.append(
                target_detection
            )
        for track_id, track in self.tracks.items():
            if track_id not in used_track_ids:
                track["positions"].clear()
        self.remove_expired_tracks()

        self.localized_publisher.publish(
            output_message
        )

        self.frame_count += 1

        if (
            self.frame_count
            % self.log_every_n_frames
            == 0
        ):
            self.get_logger().info(
                "五帧中值定位结果："
                f"{valid_position_count}/"
                f"{len(output_message.detections)}"
                "个目标具有稳定位置，"
                f"活动跟踪数量={len(self.tracks)}，"
                f"滤波窗口={self.position_filter_window}，"
                f"frame_id={self.target_frame}"
            )


def main(args=None):
    rclpy.init(args=args)

    node = VisionLocalizerNode()

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
