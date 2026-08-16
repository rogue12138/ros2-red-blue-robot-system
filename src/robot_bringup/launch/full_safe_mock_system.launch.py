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

    deepseek_parameters = os.path.join(
        bringup_directory,
        "config",
        "deepseek_params.yaml",
    )

    task_plan_parameters = os.path.join(
        bringup_directory,
        "config",
        "task_plan_params.yaml",
    )

    safety_parameters = os.path.join(
        bringup_directory,
        "config",
        "safety_params.yaml",
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
            executable="color_detector_node",
            name="color_detector_node",
            output="screen",
            parameters=[
                vision_parameters
            ],
        ),
        Node(
            package="deepseek_bridge",
            executable=(
                "deepseek_task_plan_node"
            ),
            name="deepseek_task_plan_node",
            output="screen",
            parameters=[
                deepseek_parameters
            ],
        ),
        Node(
            package="task_manager",
            executable=(
                "task_plan_manager_node"
            ),
            name="task_plan_manager_node",
            output="screen",
            parameters=[
                task_plan_parameters
            ],
        ),
        Node(
            package="task_manager",
            executable="mock_navigation_node",
            name="mock_navigation_node",
            output="screen",
            parameters=[
                safety_parameters
            ],
        ),
        Node(
            package="safety_controller",
            executable="mock_laser_node",
            name="mock_laser_node",
            output="screen",
            parameters=[
                safety_parameters
            ],
        ),
        Node(
            package="safety_controller",
            executable="safety_filter_node",
            name="safety_filter_node",
            output="screen",
            parameters=[
                safety_parameters
            ],
        ),
    ]

    return LaunchDescription(nodes)
