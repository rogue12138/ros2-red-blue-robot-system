#!/usr/bin/env python3

import math

import rclpy

from geometry_msgs.msg import Point
from rclpy.duration import Duration
from rclpy.node import Node
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray

from robot_interfaces.msg import ObjectDetectionArray


class VisionMarkerNode(Node):

    def __init__(self):
        super().__init__("vision_marker_node")

        self.declare_parameter(
            "input_topic",
            "/vision/localized_detections",
        )
        self.declare_parameter(
            "output_topic",
            "/vision/markers",
        )

        self.declare_parameter(
            "box_size_m",
            0.10,
        )
        self.declare_parameter(
            "line_width_m",
            0.01,
        )
        self.declare_parameter(
            "text_height_m",
            0.05,
        )
        self.declare_parameter(
            "text_offset_m",
            0.10,
        )
        self.declare_parameter(
            "marker_lifetime_sec",
            1.0,
        )

        self.declare_parameter(
            "track_match_distance_m",
            0.50,
        )
        self.declare_parameter(
            "track_timeout_frames",
            10,
        )
        self.declare_parameter(
            "log_every_n_frames",
            30,
        )

        self.input_topic = str(
            self.get_parameter(
                "input_topic"
            ).value
        )
        self.output_topic = str(
            self.get_parameter(
                "output_topic"
            ).value
        )

        self.box_size_m = float(
            self.get_parameter(
                "box_size_m"
            ).value
        )
        self.line_width_m = float(
            self.get_parameter(
                "line_width_m"
            ).value
        )
        self.text_height_m = float(
            self.get_parameter(
                "text_height_m"
            ).value
        )
        self.text_offset_m = float(
            self.get_parameter(
                "text_offset_m"
            ).value
        )
        self.marker_lifetime_sec = float(
            self.get_parameter(
                "marker_lifetime_sec"
            ).value
        )

        self.track_match_distance_m = float(
            self.get_parameter(
                "track_match_distance_m"
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

        if self.box_size_m <= 0.0:
            raise ValueError(
                "box_size_m必须大于0"
            )

        if self.line_width_m <= 0.0:
            raise ValueError(
                "line_width_m必须大于0"
            )

        if self.text_height_m <= 0.0:
            raise ValueError(
                "text_height_m必须大于0"
            )

        if self.marker_lifetime_sec <= 0.0:
            raise ValueError(
                "marker_lifetime_sec必须大于0"
            )

        if self.track_match_distance_m <= 0.0:
            raise ValueError(
                "track_match_distance_m必须大于0"
            )

        if self.track_timeout_frames < 1:
            raise ValueError(
                "track_timeout_frames必须大于0"
            )

        if self.log_every_n_frames < 1:
            self.log_every_n_frames = 1

        self.frame_count = 0
        self.tracks = {}
        self.next_track_id = 1

        self.marker_lifetime = Duration(
            seconds=self.marker_lifetime_sec
        ).to_msg()

        self.marker_publisher = (
            self.create_publisher(
                MarkerArray,
                self.output_topic,
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
            "视觉MarkerArray节点已启动"
        )

    def create_track(
        self,
        detection,
    ):
        track_id = self.next_track_id
        self.next_track_id += 1

        self.tracks[track_id] = {
            "class_name": detection.class_name,
            "x": float(detection.position.x),
            "y": float(detection.position.y),
            "z": float(detection.position.z),
            "last_seen_frame": self.frame_count,
        }

        return track_id

    def match_track(
        self,
        detection,
        used_track_ids,
    ):
        best_track_id = None
        best_distance = None

        for track_id, track in self.tracks.items():
            if track_id in used_track_ids:
                continue

            if (
                track["class_name"]
                != detection.class_name
            ):
                continue

            distance = math.sqrt(
                (
                    float(detection.position.x)
                    - track["x"]
                )
                ** 2
                + (
                    float(detection.position.y)
                    - track["y"]
                )
                ** 2
                + (
                    float(detection.position.z)
                    - track["z"]
                )
                ** 2
            )

            if (
                distance
                > self.track_match_distance_m
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
                detection
            )

        used_track_ids.add(best_track_id)

        track = self.tracks[best_track_id]
        track["x"] = float(
            detection.position.x
        )
        track["y"] = float(
            detection.position.y
        )
        track["z"] = float(
            detection.position.z
        )
        track["last_seen_frame"] = (
            self.frame_count
        )

        return best_track_id

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

    def set_marker_color(
        self,
        marker,
        class_name,
    ):
        marker.color.a = 1.0

        if class_name == "red_cube":
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
        elif class_name == "blue_cube":
            marker.color.r = 0.0
            marker.color.g = 0.0
            marker.color.b = 1.0
        else:
            marker.color.r = 1.0
            marker.color.g = 1.0
            marker.color.b = 1.0

    def create_point(
        self,
        x,
        y,
        z,
    ):
        point = Point()
        point.x = float(x)
        point.y = float(y)
        point.z = float(z)

        return point

    def create_line_marker(
        self,
        header,
        detection,
        track_id,
    ):
        marker = Marker()

        marker.header = header
        marker.ns = "vision_boxes"
        marker.id = track_id * 2
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD

        marker.pose.orientation.w = 1.0

        marker.scale.x = self.line_width_m

        self.set_marker_color(
            marker,
            detection.class_name,
        )

        marker.lifetime = self.marker_lifetime

        center_x = float(
            detection.position.x
        )
        center_y = float(
            detection.position.y
        )
        center_z = float(
            detection.position.z
        )

        half_size = self.box_size_m / 2.0

        marker.points = [
            self.create_point(
                center_x - half_size,
                center_y - half_size,
                center_z,
            ),
            self.create_point(
                center_x + half_size,
                center_y - half_size,
                center_z,
            ),
            self.create_point(
                center_x + half_size,
                center_y + half_size,
                center_z,
            ),
            self.create_point(
                center_x - half_size,
                center_y + half_size,
                center_z,
            ),
            self.create_point(
                center_x - half_size,
                center_y - half_size,
                center_z,
            ),
        ]

        return marker

    def create_text_marker(
        self,
        header,
        detection,
        track_id,
    ):
        marker = Marker()

        marker.header = header
        marker.ns = "vision_labels"
        marker.id = track_id * 2 + 1
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD

        marker.pose.position.x = float(
            detection.position.x
        )
        marker.pose.position.y = float(
            detection.position.y
        )
        marker.pose.position.z = (
            float(detection.position.z)
            + self.text_offset_m
        )
        marker.pose.orientation.w = 1.0

        marker.scale.z = self.text_height_m

        self.set_marker_color(
            marker,
            detection.class_name,
        )

        marker.lifetime = self.marker_lifetime

        marker.text = (
            f"ID:{track_id} "
            f"{detection.class_name}\n"
            f"conf:{detection.confidence:.2f}\n"
            "xyz:"
            f"({detection.position.x:.3f},"
            f"{detection.position.y:.3f},"
            f"{detection.position.z:.3f})"
        )

        return marker

    def create_delete_all_marker(
        self,
        header,
    ):
        marker = Marker()

        marker.header = header
        marker.ns = "vision_clear"
        marker.id = 0
        marker.action = Marker.DELETEALL

        return marker

    def detection_callback(self, message):
        marker_array = MarkerArray()

        marker_array.markers.append(
            self.create_delete_all_marker(
                message.header
            )
        )

        used_track_ids = set()
        valid_detection_count = 0

        for detection in message.detections:
            if not detection.has_position:
                continue

            track_id = self.match_track(
                detection,
                used_track_ids,
            )

            marker_array.markers.append(
                self.create_line_marker(
                    message.header,
                    detection,
                    track_id,
                )
            )

            marker_array.markers.append(
                self.create_text_marker(
                    message.header,
                    detection,
                    track_id,
                )
            )

            valid_detection_count += 1

        self.remove_expired_tracks()

        self.marker_publisher.publish(
            marker_array
        )

        self.frame_count += 1

        if (
            self.frame_count
            % self.log_every_n_frames
            == 0
        ):
            self.get_logger().info(
                "MarkerArray发布结果："
                f"目标数量={valid_detection_count}，"
                f"Marker数量={len(marker_array.markers)}，"
                f"活动ID数量={len(self.tracks)}，"
                f"frame_id={message.header.frame_id}"
            )


def main(args=None):
    rclpy.init(args=args)

    node = VisionMarkerNode()

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
