#!/usr/bin/env python3

import os

from ament_index_python.packages import (
    get_package_share_directory,
)
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import (
    PythonLaunchDescriptionSource,
)
from launch_ros.actions import Node


def generate_launch_description():
    bringup_directory = (
        get_package_share_directory(
            "robot_bringup"
        )
    )

    vision_launch_file = os.path.join(
        bringup_directory,
        "launch",
        "vision_hybrid_homography_mock.launch.py",
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

    vision_system = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            vision_launch_file
        )
    )

    deepseek_node = Node(
        package="deepseek_bridge",
        executable="deepseek_task_plan_node",
        name="deepseek_task_plan_node",
        output="screen",
        parameters=[
            deepseek_parameters
        ],
    )

    task_manager_node = Node(
        package="task_manager",
        executable="task_plan_manager_node",
        name="task_plan_manager_node",
        output="screen",
        parameters=[
            task_plan_parameters,
        ],
        remappings=[
            (
                "/vision/detections",
                "/vision/localized_detections",
            ),
        ],
    )

    navigation_node = Node(
        package="task_manager",
        executable="mock_navigation_node",
        name="mock_navigation_node",
        output="screen",
        parameters=[
            safety_parameters
        ],
    )

    laser_node = Node(
        package="safety_controller",
        executable="mock_laser_node",
        name="mock_laser_node",
        output="screen",
        parameters=[
            safety_parameters
        ],
    )

    safety_node = Node(
        package="safety_controller",
        executable="safety_filter_node",
        name="safety_filter_node",
        output="screen",
        parameters=[
            safety_parameters
        ],
    )

    return LaunchDescription(
        [
            vision_system,
            deepseek_node,
            task_manager_node,
            navigation_node,
            laser_node,
            safety_node,
        ]
    )
