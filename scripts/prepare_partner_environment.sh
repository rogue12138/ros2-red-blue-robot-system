#!/usr/bin/env bash

set -eo pipefail
SCRIPT_DIRECTORY="$(
    cd "$(dirname "${BASH_SOURCE[0]}")"
    pwd
)"

WORKSPACE_ROOT="$(
    cd "$SCRIPT_DIRECTORY/.."
    pwd
)"

MODEL_FILE="$WORKSPACE_ROOT/models/final/red_blue_yolo11n_manual180_960_best.pt"
VENV_DIRECTORY="$WORKSPACE_ROOT/venvs/yolo_cpu"
REQUIREMENTS_FILE="$WORKSPACE_ROOT/handoff_metadata/yolo_requirements_lock.txt"

echo "工作空间：$WORKSPACE_ROOT"

if [ ! -f /opt/ros/humble/setup.bash ]; then
    echo "错误：没有找到ROS 2 Humble"
    echo "请先安装Ubuntu 22.04和ROS 2 Humble"
    exit 1
fi

for required_command in \
python3 \
colcon \
rosdep
do
    if ! command -v "$required_command" >/dev/null 2>&1; then
        echo "错误：缺少命令 $required_command"
        exit 1
    fi
done

if [ ! -s "$MODEL_FILE" ]; then
    echo "错误：没有找到训练完成的模型"
    echo "$MODEL_FILE"
    exit 1
fi

echo "训练模型：正常"

if [ ! -d "$VENV_DIRECTORY" ]; then
    echo "正在创建YOLO虚拟环境"

    python3 -m venv \
    --system-site-packages \
    "$VENV_DIRECTORY"
else
    echo "YOLO虚拟环境：已经存在"
fi

source "$VENV_DIRECTORY/bin/activate"

python3 -m pip install \
--upgrade pip

if [ -s "$REQUIREMENTS_FILE" ]; then
    echo "正在按锁定清单安装Python依赖"

    if ! python3 -m pip install \
    -r "$REQUIREMENTS_FILE"
    then
        echo "完整锁定清单安装失败，改为安装核心依赖"

        python3 -m pip install \
        "numpy<2" \
        "ultralytics==8.4.120"
    fi
else
    echo "没有找到锁定清单，安装核心依赖"

    python3 -m pip install \
    "numpy<2" \
    "ultralytics==8.4.120"
fi

source /opt/ros/humble/setup.bash

cd "$WORKSPACE_ROOT"

echo "正在安装ROS包依赖"

rosdep install \
--from-paths src \
--ignore-src \
-r \
-y

echo "正在构建ROS工作空间"

colcon build \
--symlink-install

source "$WORKSPACE_ROOT/install/setup.bash"

echo "正在检查Python兼容性"

python3 -c "
import cv2
import numpy
import torch
import ultralytics
from cv_bridge import CvBridge

print('NumPy:', numpy.__version__)
print('OpenCV:', cv2.__version__)
print('PyTorch:', torch.__version__)
print('Ultralytics:', ultralytics.__version__)
print('CUDA:', torch.cuda.is_available())
print('cv_bridge:', type(CvBridge()).__name__)
"

echo "正在检查ROS节点入口"

ros2 pkg executables vision_detector |
grep yolo_detector_node

ros2 pkg executables vision_detector |
grep detection_fusion_node

ros2 pkg executables deepseek_bridge |
grep deepseek_task_plan_node

ros2 pkg executables task_manager |
grep task_plan_manager_node

ros2 pkg executables safety_controller |
grep safety_filter_node

FINAL_LAUNCH="$(
    ros2 pkg prefix robot_bringup
)/share/robot_bringup/launch/full_safe_hybrid_vision_mock.launch.py"

if [ ! -s "$FINAL_LAUNCH" ]; then
    echo "错误：没有找到最终完整启动文件"
    exit 1
fi

echo
echo "环境准备和构建已经完成"
echo "训练模型不需要重新训练"
echo
echo "下一步运行："
echo "$WORKSPACE_ROOT/scripts/start_full_system.sh"
