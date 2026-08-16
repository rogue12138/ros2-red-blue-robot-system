#!/usr/bin/env python3

import math

from collections import deque

import numpy as np
import rclpy

from rclpy.node import Node

from robot_interfaces.msg import ObjectDetection
from robot_interfaces.msg import ObjectDetectionArray


class HomographyLocalizerNode(Node):

    def __init__(self):
        super().__init__(
            "homography_localizer_node"
        )

        self.declare_parameter(
            "detections_topic",
            "/vision/detections",
        )
        self.declare_parameter(
            "output_topic",
            "/vision/localized_detections",
        )
        self.declare_parameter(
            "output_frame",
            "base_link",
        )
        self.declare_parameter(
            "table_height_m",
            0.0,
        )

        self.declare_parameter(
            "homography_matrix",
            [
                0.0,
                -0.0025,
                1.5,
                -0.002,
                0.0,
                0.639,
                0.0,
                0.0,
                1.0,
            ],
        )

        self.declare_parameter(
            "minimum_denominator",
            1.0e-8,
        )
        self.declare_parameter(
            "maximum_coordinate_abs_m",
            10.0,
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
        self.declare_parameter(
            "log_every_n_frames",
            30,
        )

        self.detections_topic = str(
            self.get_parameter(
                "detections_topic"
            ).value
        )
        self.output_topic = str(
            self.get_parameter(
                "output_topic"
            ).value
        )
        self.output_frame = str(
            self.get_parameter(
                "output_frame"
            ).value
        )
        self.table_height_m = float(
            self.get_parameter(
                "table_height_m"
            ).value
        )

        homography_values = list(
            self.get_parameter(
                "homography_matrix"
            ).value
        )

        self.minimum_denominator = float(
            self.get_parameter(
                "minimum_denominator"
            ).value
        )
        self.maximum_coordinate_abs_m = float(
            self.get_parameter(
                "maximum_coordinate_abs_m"
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
        self.log_every_n_frames = int(
            self.get_parameter(
                "log_every_n_frames"
            ).value
        )

        if not self.output_frame:
            raise ValueError(
                "output_frame不能为空"
            )

        if len(homography_values) != 9:
            raise ValueError(
                "homography_matrix必须包含9个元素"
            )

        if self.minimum_denominator <= 0.0:
            raise ValueError(
                "minimum_denominator必须大于0"
            )

        if self.maximum_coordinate_abs_m <= 0.0:
            raise ValueError(
                "maximum_coordinate_abs_m必须大于0"
            )

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

        if self.log_every_n_frames < 1:
            self.log_every_n_frames = 1

        self.homography_matrix = np.asarray(
            homography_values,
            dtype=np.float64,
        ).reshape(
            (
                3,
                3,
            )
        )

        determinant = float(
            np.linalg.det(
                self.homography_matrix
            )
        )

        if abs(determinant) <= 1.0e-12:
            raise ValueError(
                "homography_matrix不可逆"
            )

        self.frame_count = 0
        self.tracks = {}
        self.next_track_id = 1

        self.localized_publisher = (
            self.create_publisher(
                ObjectDetectionArray,
                self.output_topic,
                10,
            )
        )

        self.detection_subscription = (
            self.create_subscription(
                ObjectDetectionArray,
                self.detections_topic,
                self.detection_callback,
                10,
            )
        )

        self.get_logger().info(
            "无深度单应性定位节点已启动，"
            f"output_frame={self.output_frame}，"
            f"table_height={self.table_height_m:.3f}m，"
            f"filter_window={self.position_filter_window}"
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

    def project_pixel(
        self,
        center_x,
        center_y,
    ):
        pixel_vector = np.asarray(
            [
                float(center_x),
                float(center_y),
                1.0,
            ],
            dtype=np.float64,
        )

        mapped_vector = (
            self.homography_matrix
            @ pixel_vector
        )

        denominator = float(
            mapped_vector[2]
        )

        if (
            abs(denominator)
            < self.minimum_denominator
        ):
            return None

        position_x = float(
            mapped_vector[0]
            / denominator
        )
        position_y = float(
            mapped_vector[1]
            / denominator
        )
        position_z = self.table_height_m

        if not (
            math.isfinite(position_x)
            and math.isfinite(position_y)
            and math.isfinite(position_z)
        ):
            return None

        if (
            abs(position_x)
            > self.maximum_coordinate_abs_m
            or abs(position_y)
            > self.maximum_coordinate_abs_m
        ):
            return None

        return (
            position_x,
            position_y,
            position_z,
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

    def detection_callback(self, message):
        output_message = ObjectDetectionArray()
        output_message.header = message.header
        output_message.header.frame_id = (
            self.output_frame
        )

        valid_position_count = 0
        used_track_ids = set()

        for source_detection in message.detections:
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

            projected_position = (
                self.project_pixel(
                    center_x,
                    center_y,
                )
            )

            filtered_position = self.update_track(
                track_id,
                center_x,
                center_y,
                projected_position,
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
                "单应性五帧中值定位："
                f"{valid_position_count}/"
                f"{len(output_message.detections)}"
                "个目标具有稳定位置，"
                f"活动跟踪数量={len(self.tracks)}，"
                f"frame_id={self.output_frame}"
            )


def main(args=None):
    rclpy.init(args=args)

    node = HomographyLocalizerNode()

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
