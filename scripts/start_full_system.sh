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

VENV_DIRECTORY="$WORKSPACE_ROOT/venvs/yolo_cpu"
MODEL_FILE="$WORKSPACE_ROOT/models/final/red_blue_yolo11n_manual180_960_best.pt"

if [ ! -f /opt/ros/humble/setup.bash ]; then
    echo "错误：没有找到ROS 2 Humble"
    exit 1
fi

if [ ! -f "$VENV_DIRECTORY/bin/activate" ]; then
    echo "错误：YOLO环境尚未建立"
    echo "请先运行："
    echo "$WORKSPACE_ROOT/scripts/prepare_partner_environment.sh"
    exit 1
fi

if [ ! -s "$MODEL_FILE" ]; then
    echo "错误：训练模型不存在"
    echo "$MODEL_FILE"
    exit 1
fi

if [ ! -f "$WORKSPACE_ROOT/install/setup.bash" ]; then
    echo "错误：工作空间尚未构建"
    echo "请先运行："
    echo "$WORKSPACE_ROOT/scripts/prepare_partner_environment.sh"
    exit 1
fi

source "$VENV_DIRECTORY/bin/activate"
source /opt/ros/humble/setup.bash
source "$WORKSPACE_ROOT/install/setup.bash"

if [ -z "${DEEPSEEK_API_KEY:-}" ]; then
    read -r -s \
    -p "请输入DeepSeek API密钥：" \
    DEEPSEEK_API_KEY

    echo

    if [ -z "$DEEPSEEK_API_KEY" ]; then
        echo "错误：API密钥不能为空"
        exit 1
    fi

    export DEEPSEEK_API_KEY
fi

echo "工作空间：$WORKSPACE_ROOT"
echo "训练模型：已经加载"
echo "DeepSeek密钥：已经加载"
echo "正在启动完整安全混合视觉系统"

exec ros2 launch \
robot_bringup \
full_safe_hybrid_vision_mock.launch.py
