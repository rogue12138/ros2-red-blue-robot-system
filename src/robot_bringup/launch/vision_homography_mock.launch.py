#!/usr/bin/env python3

import os

from ament_index_python.packages import (
    get_package_share_directory,
)
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    bringup_directory = (
        get_package_share_directory(
            "robot_bringup"
        )
    )

    homography_parameters = os.path.join(
        bringup_directory,
        "config",
        "homography_params.yaml",
    )

    nodes = [
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
            executable="homography_localizer_node",
            name="homography_localizer_node",
            output="screen",
            parameters=[
                homography_parameters
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
