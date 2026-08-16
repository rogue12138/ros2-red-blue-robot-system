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

    vision_parameters = os.path.join(
        bringup_directory,
        "config",
        "vision_params.yaml",
    )

    hybrid_parameters = os.path.join(
        bringup_directory,
        "config",
        "hybrid_vision_params.yaml",
    )

    homography_parameters = os.path.join(
        bringup_directory,
        "config",
        "hybrid_homography_params.yaml",
    )

    random_camera_node = Node(
        package="vision_detector",
        executable="random_scene_camera_node",
        name="random_scene_camera_node",
        output="screen",
        parameters=[hybrid_parameters],
    )

    hsv_detector_node = Node(
        package="vision_detector",
        executable="red_blue_detector_node",
        name="red_blue_detector_node",
        output="screen",
        parameters=[vision_parameters],
    )

    yolo_detector_node = Node(
        package="vision_detector",
        executable="yolo_detector_node",
        name="yolo_detector_node",
        output="screen",
        parameters=[hybrid_parameters],
    )

    fusion_node = Node(
        package="vision_detector",
        executable="detection_fusion_node",
        name="detection_fusion_node",
        output="screen",
        parameters=[hybrid_parameters],
    )

    homography_node = Node(
        package="vision_detector",
        executable="homography_localizer_node",
        name="homography_localizer_node",
        output="screen",
        parameters=[homography_parameters],
    )

    marker_node = Node(
        package="vision_detector",
        executable="vision_marker_node",
        name="vision_marker_node",
        output="screen",
    )

    return LaunchDescription(
        [
            random_camera_node,
            hsv_detector_node,
            yolo_detector_node,
            fusion_node,
            homography_node,
            marker_node,
        ]
    )
