#!/usr/bin/env python3

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    bringup_directory = get_package_share_directory(
        "robot_bringup"
    )

    vision_parameters = os.path.join(
        bringup_directory,
        "config",
        "vision_params.yaml",
    )

    deepseek_parameters = os.path.join(
        bringup_directory,
        "config",
        "deepseek_params.yaml",
    )

    mock_camera_node = Node(
        package="vision_detector",
        executable="mock_camera_node",
        name="mock_camera_node",
        output="screen",
    )

    color_detector_node = Node(
        package="vision_detector",
        executable="color_detector_node",
        name="color_detector_node",
        output="screen",
        parameters=[vision_parameters],
    )

    deepseek_command_node = Node(
        package="deepseek_bridge",
        executable="deepseek_command_node",
        name="deepseek_command_node",
        output="screen",
        parameters=[deepseek_parameters],
    )

    task_manager_node = Node(
        package="task_manager",
        executable="task_manager_node",
        name="task_manager_node",
        output="screen",
    )

    return LaunchDescription([
        mock_camera_node,
        color_detector_node,
        deepseek_command_node,
        task_manager_node,
    ])
