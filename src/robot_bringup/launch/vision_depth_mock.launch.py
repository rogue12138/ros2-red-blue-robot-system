#!/usr/bin/env python3

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    nodes = [
        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            name="vision_camera_static_tf",
            output="screen",
            arguments=[
                "--x", "0.20",
                "--y", "0.0",
                "--z", "0.50",
                "--roll", "-1.57079632679",
                "--pitch", "0.0",
                "--yaw", "-1.57079632679",
                "--frame-id", "base_link",
                "--child-frame-id", "camera_frame",
            ],
        ),
        Node(
            package="vision_detector",
            executable="mock_camera_node",
            name="mock_camera_node",
            output="screen",
        ),
        Node(
            package="vision_detector",
            executable="red_blue_detector_node",
            name="red_blue_detector_node",
            output="screen",
            parameters=[
                {
                    "minimum_area": 500.0,
                    "maximum_area_ratio": 0.20,
                    "minimum_aspect_ratio": 0.45,
                    "maximum_aspect_ratio": 2.20,
                    "minimum_fill_ratio": 0.50,
                    "morph_kernel_size": 5,
                    "log_every_n_frames": 10,
                }
            ],
        ),
        Node(
            package="vision_detector",
            executable="mock_depth_camera_node",
            name="mock_depth_camera_node",
            output="screen",
        ),
        Node(
            package="vision_detector",
            executable="vision_localizer_node",
            name="vision_localizer_node",
            output="screen",
            parameters=[
                {
                    "target_frame": "base_link",
                    "position_filter_window": 5,
                    "log_every_n_frames": 10,
                }
            ],
        ),
        Node(
            package="vision_detector",
            executable="vision_marker_node",
            name="vision_marker_node",
            output="screen",
        ),
    ]

    return LaunchDescription(nodes)
