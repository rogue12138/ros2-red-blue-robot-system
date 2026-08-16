from setuptools import find_packages, setup

package_name = 'vision_detector'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ros',
    maintainer_email='ros@todo.todo',
    description='TODO: Package description',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        "console_scripts": [
            "mock_camera_node = vision_detector.mock_camera_node:main",
            "color_detector_node = vision_detector.color_detector_node:main",
            "red_blue_detector_node = vision_detector.red_blue_detector_node:main",
            "random_scene_camera_node = vision_detector.random_scene_camera_node:main",
            "dataset_capture_node = vision_detector.dataset_capture_node:main",
            "mock_depth_camera_node = vision_detector.mock_depth_camera_node:main",
            "vision_localizer_node = vision_detector.vision_localizer_node:main",
            "homography_localizer_node = vision_detector.homography_localizer_node:main",
            "vision_marker_node = vision_detector.vision_marker_node:main",
            "vision_acceptance_node = vision_detector.vision_acceptance_node:main",
            "position_stability_acceptance_node = vision_detector.position_stability_acceptance_node:main",
            "yolo_detector_node = vision_detector.yolo_detector_node:main",
            "detection_fusion_node = vision_detector.detection_fusion_node:main",
    ],
    },
)
