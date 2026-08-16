#!/usr/bin/env python3

import json
import random

import cv2
import numpy as np
import rclpy

from cv_bridge import CvBridge
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image

from robot_interfaces.msg import ObjectDetection
from robot_interfaces.msg import ObjectDetectionArray


class RandomSceneCameraNode(Node):

    def __init__(self):
        super().__init__(
            "random_scene_camera_node"
        )

        self.declare_parameter(
            "image_width",
            640,
        )
        self.declare_parameter(
            "image_height",
            480,
        )
        self.declare_parameter(
            "fps",
            5.0,
        )
        self.declare_parameter(
            "scene_change_interval_sec",
            2.0,
        )
        self.declare_parameter(
            "minimum_total_cubes",
            1,
        )
        self.declare_parameter(
            "maximum_total_cubes",
            8,
        )
        self.declare_parameter(
            "fixed_red_count",
            -1,
        )
        self.declare_parameter(
            "fixed_blue_count",
            -1,
        )
        self.declare_parameter(
            "minimum_cube_size",
            36,
        )
        self.declare_parameter(
            "maximum_cube_size",
            82,
        )
        self.declare_parameter(
            "fixed_obstacle_count",
            -1,
        )
        self.declare_parameter(
            "maximum_obstacles",
            2,
        )
        self.declare_parameter(
            "overlap_probability",
            0.15,
        )
        self.declare_parameter(
            "random_seed",
            42,
        )

        self.image_width = int(
            self.get_parameter(
                "image_width"
            ).value
        )
        self.image_height = int(
            self.get_parameter(
                "image_height"
            ).value
        )
        self.fps = float(
            self.get_parameter(
                "fps"
            ).value
        )
        self.scene_change_interval_sec = float(
            self.get_parameter(
                "scene_change_interval_sec"
            ).value
        )

        self.minimum_total_cubes = int(
            self.get_parameter(
                "minimum_total_cubes"
            ).value
        )
        self.maximum_total_cubes = int(
            self.get_parameter(
                "maximum_total_cubes"
            ).value
        )
        self.fixed_red_count = int(
            self.get_parameter(
                "fixed_red_count"
            ).value
        )
        self.fixed_blue_count = int(
            self.get_parameter(
                "fixed_blue_count"
            ).value
        )

        self.minimum_cube_size = int(
            self.get_parameter(
                "minimum_cube_size"
            ).value
        )
        self.maximum_cube_size = int(
            self.get_parameter(
                "maximum_cube_size"
            ).value
        )

        self.fixed_obstacle_count = int(
            self.get_parameter(
                "fixed_obstacle_count"
            ).value
        )
        self.maximum_obstacles = int(
            self.get_parameter(
                "maximum_obstacles"
            ).value
        )
        self.overlap_probability = float(
            self.get_parameter(
                "overlap_probability"
            ).value
        )

        random_seed = int(
            self.get_parameter(
                "random_seed"
            ).value
        )

        if random_seed < 0:
            self.random_generator = (
                random.Random()
            )
        else:
            self.random_generator = (
                random.Random(random_seed)
            )

        self.validate_parameters()

        self.bridge = CvBridge()
        self.scene_index = 0
        self.current_image = None
        self.current_detections = []
        self.current_scene_info = {}

        self.image_publisher = (
            self.create_publisher(
                Image,
                "/camera/image_raw",
                10,
            )
        )

        self.ground_truth_publisher = (
            self.create_publisher(
                ObjectDetectionArray,
                "/simulation/ground_truth_detections",
                10,
            )
        )

        self.scene_info_publisher = (
            self.create_publisher(
                String,
                "/simulation/scene_info",
                10,
            )
        )

        timer_period = 1.0 / self.fps

        self.publish_timer = self.create_timer(
            timer_period,
            self.publish_scene,
        )

        self.scene_timer = self.create_timer(
            self.scene_change_interval_sec,
            self.generate_scene,
        )

        self.generate_scene()

        self.get_logger().info(
            "随机红蓝仿真摄像头已启动"
        )

    def validate_parameters(self):
        if self.image_width < 320:
            self.image_width = 320

        if self.image_height < 240:
            self.image_height = 240

        if self.fps <= 0.0:
            self.fps = 1.0

        if self.scene_change_interval_sec <= 0.0:
            self.scene_change_interval_sec = 1.0

        if self.minimum_total_cubes < 0:
            self.minimum_total_cubes = 0

        if (
            self.maximum_total_cubes
            < self.minimum_total_cubes
        ):
            self.maximum_total_cubes = (
                self.minimum_total_cubes
            )

        if self.minimum_cube_size < 12:
            self.minimum_cube_size = 12

        if (
            self.maximum_cube_size
            < self.minimum_cube_size
        ):
            self.maximum_cube_size = (
                self.minimum_cube_size
            )

        self.maximum_obstacles = max(
            0,
            self.maximum_obstacles,
        )

        self.overlap_probability = min(
            1.0,
            max(
                0.0,
                self.overlap_probability,
            ),
        )

    def select_cube_counts(self):
        if (
            self.fixed_red_count >= 0
            and self.fixed_blue_count >= 0
        ):
            return (
                self.fixed_red_count,
                self.fixed_blue_count,
            )

        total_count = (
            self.random_generator.randint(
                self.minimum_total_cubes,
                self.maximum_total_cubes,
            )
        )

        red_count = (
            self.random_generator.randint(
                0,
                total_count,
            )
        )
        blue_count = (
            total_count - red_count
        )

        return red_count, blue_count

    def select_obstacle_count(self):
        if self.fixed_obstacle_count >= 0:
            return self.fixed_obstacle_count

        return self.random_generator.randint(
            0,
            self.maximum_obstacles,
        )

    def create_background(self):
        image = np.zeros(
            (
                self.image_height,
                self.image_width,
                3,
            ),
            dtype=np.uint8,
        )

        horizon = self.random_generator.randint(
            int(self.image_height * 0.20),
            int(self.image_height * 0.35),
        )

        ceiling_color = (
            self.random_generator.randint(
                145,
                185,
            )
        )
        floor_color = (
            self.random_generator.randint(
                165,
                205,
            )
        )

        image[:horizon, :] = (
            ceiling_color,
            ceiling_color,
            ceiling_color,
        )
        image[horizon:, :] = (
            floor_color,
            floor_color,
            floor_color,
        )

        vanishing_x = (
            self.random_generator.randint(
                int(self.image_width * 0.35),
                int(self.image_width * 0.65),
            )
        )

        for x_position in range(
            0,
            self.image_width,
            80,
        ):
            cv2.line(
                image,
                (x_position, self.image_height),
                (vanishing_x, horizon),
                (130, 130, 130),
                1,
            )

        for y_position in range(
            horizon + 45,
            self.image_height,
            55,
        ):
            cv2.line(
                image,
                (0, y_position),
                (self.image_width, y_position),
                (140, 140, 140),
                1,
            )

        self.draw_drop_zones(image)
        self.draw_hard_negative_regions(image)

        return image, horizon

    def draw_drop_zones(self, image):
        zone_width = int(
            self.image_width * 0.16
        )
        zone_height = int(
            self.image_height * 0.10
        )
        bottom_y = (
            self.image_height
            - zone_height
            - 12
        )

        zone_colors = [
            (120, 180, 180),
            (180, 180, 120),
            (180, 140, 180),
        ]

        for index, zone_name in enumerate(
            ["A", "B", "C"]
        ):
            x_position = (
                20
                + index
                * (zone_width + 25)
            )

            cv2.rectangle(
                image,
                (x_position, bottom_y),
                (
                    x_position + zone_width,
                    bottom_y + zone_height,
                ),
                zone_colors[index],
                2,
            )

            cv2.putText(
                image,
                f"ZONE {zone_name}",
                (
                    x_position + 8,
                    bottom_y + 28,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                zone_colors[index],
                2,
                cv2.LINE_AA,
            )

    def draw_hard_negative_regions(
        self,
        image,
    ):
        if (
            self.random_generator.random()
            < 0.65
        ):
            wall_y = self.random_generator.randint(
                25,
                80,
            )

            cv2.rectangle(
                image,
                (15, wall_y),
                (
                    int(self.image_width * 0.45),
                    wall_y + 55,
                ),
                (45, 45, 115),
                -1,
            )

            for x_position in range(
                20,
                int(self.image_width * 0.45),
                38,
            ):
                cv2.line(
                    image,
                    (x_position, wall_y),
                    (x_position, wall_y + 55),
                    (65, 65, 145),
                    2,
                )

        if (
            self.random_generator.random()
            < 0.45
        ):
            panel_x = int(
                self.image_width * 0.62
            )
            panel_y = self.random_generator.randint(
                30,
                100,
            )

            cv2.rectangle(
                image,
                (panel_x, panel_y),
                (
                    self.image_width - 15,
                    panel_y + 35,
                ),
                (125, 65, 25),
                -1,
            )

    def draw_obstacles(
        self,
        image,
        obstacle_count,
        horizon,
    ):
        obstacle_boxes = []

        for _ in range(obstacle_count):
            width = self.random_generator.randint(
                45,
                100,
            )
            height = self.random_generator.randint(
                50,
                130,
            )

            x_position = (
                self.random_generator.randint(
                    10,
                    max(
                        10,
                        self.image_width
                        - width
                        - 10,
                    ),
                )
            )
            y_position = (
                self.random_generator.randint(
                    horizon,
                    max(
                        horizon,
                        self.image_height
                        - height
                        - 25,
                    ),
                )
            )

            obstacle_color = (
                self.random_generator.randint(
                    70,
                    125,
                )
            )

            cv2.rectangle(
                image,
                (x_position, y_position),
                (
                    x_position + width,
                    y_position + height,
                ),
                (
                    obstacle_color,
                    obstacle_color,
                    obstacle_color,
                ),
                -1,
            )

            cv2.rectangle(
                image,
                (x_position, y_position),
                (
                    x_position + width,
                    y_position + height,
                ),
                (40, 40, 40),
                2,
            )

            obstacle_boxes.append(
                (
                    x_position,
                    y_position,
                    x_position + width,
                    y_position + height,
                )
            )

        return obstacle_boxes

    def boxes_overlap(
        self,
        first_box,
        second_box,
    ):
        first_left = first_box[0]
        first_top = first_box[1]
        first_right = first_box[2]
        first_bottom = first_box[3]

        second_left = second_box[0]
        second_top = second_box[1]
        second_right = second_box[2]
        second_bottom = second_box[3]

        return not (
            first_right < second_left
            or second_right < first_left
            or first_bottom < second_top
            or second_bottom < first_top
        )

    def choose_cube_box(
        self,
        size,
        depth,
        horizon,
        existing_boxes,
    ):
        for _ in range(80):
            x_position = (
                self.random_generator.randint(
                    12,
                    max(
                        12,
                        self.image_width
                        - size
                        - depth
                        - 12,
                    ),
                )
            )
            y_position = (
                self.random_generator.randint(
                    horizon + depth + 5,
                    max(
                        horizon + depth + 5,
                        self.image_height
                        - size
                        - 25,
                    ),
                )
            )

            candidate_box = (
                x_position,
                y_position - depth,
                x_position + size + depth,
                y_position + size,
            )

            allow_overlap = (
                self.random_generator.random()
                < self.overlap_probability
            )

            if allow_overlap:
                return candidate_box

            overlaps_existing = any(
                self.boxes_overlap(
                    candidate_box,
                    existing_box,
                )
                for existing_box
                in existing_boxes
            )

            if not overlaps_existing:
                return candidate_box

        return candidate_box

    def adjust_color(
        self,
        base_color,
        difference,
    ):
        return tuple(
            int(
                min(
                    255,
                    max(
                        0,
                        channel + difference,
                    ),
                )
            )
            for channel in base_color
        )

    def draw_cube(
        self,
        image,
        class_name,
        bounding_box,
    ):
        left = bounding_box[0]
        top = bounding_box[1]
        right = bounding_box[2]
        bottom = bounding_box[3]

        depth = max(
            6,
            int((bottom - top) * 0.16),
        )

        front_left = left
        front_top = top + depth
        front_right = right - depth
        front_bottom = bottom

        if class_name == "red_cube":
            base_color = (
                self.random_generator.randint(
                    0,
                    25,
                ),
                self.random_generator.randint(
                    0,
                    35,
                ),
                self.random_generator.randint(
                    205,
                    255,
                ),
            )
        else:
            base_color = (
                self.random_generator.randint(
                    205,
                    255,
                ),
                self.random_generator.randint(
                    0,
                    45,
                ),
                self.random_generator.randint(
                    0,
                    30,
                ),
            )

        top_color = self.adjust_color(
            base_color,
            25,
        )
        side_color = self.adjust_color(
            base_color,
            -35,
        )

        front_points = np.array(
            [
                [front_left, front_top],
                [front_right, front_top],
                [front_right, front_bottom],
                [front_left, front_bottom],
            ],
            dtype=np.int32,
        )

        top_points = np.array(
            [
                [front_left, front_top],
                [front_left + depth, top],
                [right, top],
                [front_right, front_top],
            ],
            dtype=np.int32,
        )

        side_points = np.array(
            [
                [front_right, front_top],
                [right, top],
                [right, bottom - depth],
                [front_right, front_bottom],
            ],
            dtype=np.int32,
        )

        cv2.fillPoly(
            image,
            [front_points],
            base_color,
        )
        cv2.fillPoly(
            image,
            [top_points],
            top_color,
        )
        cv2.fillPoly(
            image,
            [side_points],
            side_color,
        )

        cv2.polylines(
            image,
            [front_points],
            True,
            (25, 25, 25),
            2,
        )
        cv2.polylines(
            image,
            [top_points],
            True,
            (25, 25, 25),
            2,
        )
        cv2.polylines(
            image,
            [side_points],
            True,
            (25, 25, 25),
            2,
        )

    def create_detection(
        self,
        class_name,
        bounding_box,
    ):
        detection = ObjectDetection()

        detection.class_name = class_name
        detection.confidence = 1.0

        detection.x_min = int(
            bounding_box[0]
        )
        detection.y_min = int(
            bounding_box[1]
        )
        detection.x_max = int(
            bounding_box[2]
        )
        detection.y_max = int(
            bounding_box[3]
        )

        detection.has_position = False
        detection.position.x = 0.0
        detection.position.y = 0.0
        detection.position.z = 0.0

        return detection

    def generate_scene(self):
        (
            red_count,
            blue_count,
        ) = self.select_cube_counts()

        obstacle_count = (
            self.select_obstacle_count()
        )

        image, horizon = (
            self.create_background()
        )

        obstacle_boxes = self.draw_obstacles(
            image,
            obstacle_count,
            horizon,
        )

        cube_classes = (
            ["red_cube"] * red_count
            + ["blue_cube"] * blue_count
        )

        self.random_generator.shuffle(
            cube_classes
        )

        existing_boxes = list(
            obstacle_boxes
        )
        detections = []

        for class_name in cube_classes:
            size = (
                self.random_generator.randint(
                    self.minimum_cube_size,
                    self.maximum_cube_size,
                )
            )
            depth = max(
                6,
                int(size * 0.18),
            )

            bounding_box = (
                self.choose_cube_box(
                    size,
                    depth,
                    horizon,
                    existing_boxes,
                )
            )

            self.draw_cube(
                image,
                class_name,
                bounding_box,
            )

            detections.append(
                self.create_detection(
                    class_name,
                    bounding_box,
                )
            )

            existing_boxes.append(
                bounding_box
            )

        self.scene_index += 1
        self.current_image = image
        self.current_detections = detections
        self.current_scene_info = {
            "scene_index": self.scene_index,
            "red_count": red_count,
            "blue_count": blue_count,
            "total_count": (
                red_count + blue_count
            ),
            "obstacle_count": obstacle_count,
        }

        self.get_logger().info(
            "生成随机场景："
            f"scene={self.scene_index}，"
            f"red={red_count}，"
            f"blue={blue_count}，"
            f"obstacles={obstacle_count}"
        )

    def publish_scene(self):
        if self.current_image is None:
            return

        stamp = self.get_clock().now().to_msg()

        image_message = (
            self.bridge.cv2_to_imgmsg(
                self.current_image,
                encoding="bgr8",
            )
        )
        image_message.header.stamp = stamp
        image_message.header.frame_id = (
            "camera_color_optical_frame"
        )

        self.image_publisher.publish(
            image_message
        )

        ground_truth_message = (
            ObjectDetectionArray()
        )
        ground_truth_message.header = (
            image_message.header
        )
        ground_truth_message.detections = (
            self.current_detections
        )

        self.ground_truth_publisher.publish(
            ground_truth_message
        )

        scene_info_message = String()
        scene_info_message.data = json.dumps(
            self.current_scene_info,
            ensure_ascii=False,
        )

        self.scene_info_publisher.publish(
            scene_info_message
        )


def main(args=None):
    rclpy.init(args=args)

    node = RandomSceneCameraNode()

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
