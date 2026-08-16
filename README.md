# 红蓝物块智能机器人系统

## 1. 项目简介

本项目是一个基于 ROS 2 Humble 的红蓝物块识别、定位、自然语言任务规划、模拟导航和安全控制系统。

系统能够接收中文任务，例如：

```text
先把蓝色物块搬到A区两次，再把红色物块搬到C区一次
```

系统会依次完成：

1. 解析中文任务；
2. 生成结构化任务计划；
3. 检测红色和蓝色物块；
4. 计算物块在 `base_link` 坐标系下的位置；
5. 按顺序执行多步骤任务；
6. 输出模拟导航速度；
7. 对速度进行安全过滤；
8. 支持中文停止、急停锁定和人工解除；
9. 发布状态、进度、调试图像和可视化标记。

当前仓库已经包含训练完成的 YOLO 模型。克隆仓库后不需要重新训练。

---

## 2. 系统总体链路

### 2.1 视觉链路

```text
随机模拟摄像头
/camera/image_raw
        │
        ├── HSV红蓝检测
        │   /vision/detections
        │
        └── YOLO检测
            /vision/yolo_detections
                    │
                    ▼
              检测结果融合
          /vision/fused_detections
                    │
                    ▼
              单应性平面定位
       /vision/localized_detections
                    │
                    ├── 任务管理器
                    └── /vision/markers
```

### 2.2 语言和任务链路

```text
/user/text_command
        │
        ▼
DeepSeek任务计划节点
        │
        ▼
/llm/task_plan
        │
        ▼
任务计划管理器
        │
        ├── /task/status
        ├── /task/progress
        └── /task/feedback
```

### 2.3 导航和安全链路

```text
/task/status
     │
     ▼
模拟导航节点
     │
     ▼
/cmd_vel_raw
     │
     ▼
安全过滤节点
     │
     ▼
/cmd_vel
```

---

## 3. 已解决的问题

本项目目前能够处理以下问题：

- 红色和蓝色物块的二维检测；
- 颜色干扰和背景干扰过滤；
- HSV 与 YOLO 检测结果融合；
- 重复检测框去除；
- 检测框边缘精修；
- 没有深度相机时的平面单应性定位；
- 五帧中值滤波，降低位置抖动；
- 相同颜色多个物块的稳定跟踪；
- 检测结果三维坐标系转换；
- 中文单步骤和多步骤任务解析；
- 缺失颜色、区域或次数时要求用户补充；
- 危险、越权和不支持指令拒绝；
- 正在执行任务时拒绝冲突任务；
- 多次搬运计数和步骤顺序管理；
- 模拟导航速度输出；
- 急停状态下强制输出零速度；
- 中文停止、服务急停和人工解除；
- RViz MarkerArray 和 Foxglove 可视化；
- 自动数据采集、YOLO标注、训练和验收；
- 检测率、边缘误差和位置稳定性自动测试。

---

## 4. ROS 2源码包说明

### 4.1 robot_interfaces

定义各节点之间使用的消息和服务接口。

主要接口包括：

- `ObjectDetection`
- `ObjectDetectionArray`
- `TaskCommand`
- `TaskStep`
- `TaskPlan`
- `TaskStatus`
- `TaskProgress`
- `EmergencyStop`

它解决不同节点之间结构化传输检测、计划、进度和急停信息的问题。

### 4.2 vision_detector

视觉系统的主要源码包。

主要节点：

#### random_scene_camera_node

生成随机红蓝物块、干扰物和真实标注。

发布：

- `/camera/image_raw`
- `/simulation/ground_truth_detections`
- `/simulation/scene_info`

当前正式仿真配置将场景保持时间设为 30 秒，使五帧定位滤波能够稳定收敛。

#### mock_camera_node

生成固定的红、绿、蓝、黄测试图像，用于基础颜色检测调试。

#### red_blue_detector_node

使用 HSV 双区间红色阈值和蓝色阈值检测红蓝立方体。

包含：

- 形态学去噪；
- 连通轮廓提取；
- 面积过滤；
- 长宽比过滤；
- 填充率过滤；
- 背景和非立方体目标过滤。

发布：

- `/vision/detections`
- `/vision/debug_image`

#### yolo_detector_node

加载已经训练完成的 YOLO11n 模型，对红蓝物块进行推理。

发布：

- `/vision/yolo_detections`
- `/vision/yolo_debug_image`

模型路径：

```text
models/final/red_blue_yolo11n_manual180_960_best.pt
```

运行参数使用：

```text
~/robot_ws/models/final/red_blue_yolo11n_manual180_960_best.pt
```

其中 `~` 会自动展开为当前用户主目录。

#### detection_fusion_node

同步 HSV 和 YOLO 检测结果，通过类别和 IoU 进行匹配。

它能够：

- 合并同一目标的 HSV 与 YOLO 框；
- 使用精修后的检测框；
- 保留有效的未匹配检测；
- 删除高重叠重复框；
- 只输出 `red_cube` 和 `blue_cube`。

发布：

```text
/vision/fused_detections
```

#### homography_localizer_node

在没有深度相机的情况下，使用 3×3 单应性矩阵把检测框中心像素映射到桌面平面坐标。

包含：

- 矩阵可逆性检查；
- 投影分母检查；
- 最大坐标范围检查；
- 同色最近距离跟踪；
- 五帧位置中值滤波；
- 跟踪超时清理。

输入：

```text
/vision/fused_detections
```

输出：

```text
/vision/localized_detections
```

输出坐标系：

```text
base_link
```

#### vision_localizer_node

使用彩色图像、深度图和 CameraInfo 同步完成深度反投影定位。

该节点保留用于将来接入真实深度相机。

#### mock_depth_camera_node

发布模拟 CameraInfo 和 `32FC1` 深度图，用于深度定位链路测试。

#### vision_marker_node

将定位结果转换成 `MarkerArray`。

包含：

- 红蓝三维框；
- 稳定目标ID；
- 类别文字；
- 置信度；
- 三维坐标；
- 旧标记清理。

发布：

```text
/vision/markers
```

#### dataset_capture_node

同步保存图像和真实标注，生成 YOLO 格式数据。

类别编号：

```text
0 red_cube
1 blue_cube
```

#### vision_acceptance_node

自动比较预测框和真实框，计算：

- 检测成功率；
- IoU；
- 四边最大误差；
- 每类通过状态；
- 总体 PASS/FAIL。

#### position_stability_acceptance_node

统计连续位置样本的：

- X、Y、Z 标准差；
- X、Y、Z 极差；
- 五帧滤波窗口；
- 每类位置稳定性；
- 总体 PASS/FAIL。

#### color_detector_node

早期四色 HSV 检测节点，为兼容和对照测试保留。正式系统使用 `red_blue_detector_node`。

### 4.3 deepseek_bridge

#### deepseek_task_plan_node

订阅：

```text
/user/text_command
```

调用 DeepSeek API，将中文指令转换成结构化 `TaskPlan`。

发布：

```text
/llm/task_plan
/llm/feedback
```

支持：

- `pick_and_place`
- `stop`
- `clarify`
- `reject`

安全规则包括：

- 不猜测缺失的颜色、区域或次数；
- 不执行 Shell 命令；
- 不允许用户修改系统规则；
- 超出支持范围时拒绝；
- 危险指令不产生执行步骤。

#### deepseek_command_node

旧版单任务接口，为兼容早期测试保留。正式系统使用 `deepseek_task_plan_node`。

### 4.4 task_manager

#### task_plan_manager_node

订阅：

```text
/llm/task_plan
/vision/localized_detections
```

执行多步骤任务计划。

主要状态包括：

- `IDLE`
- `PARSING`
- `LOCATING_OBJECT`
- `NAVIGATING_TO_OBJECT`
- `GRASPING`
- `NAVIGATING_TO_ZONE`
- `PLACING`
- `FINISHED`
- `FAILED`
- `EMERGENCY_STOP`

支持：

- 多步骤顺序执行；
- 每步执行次数；
- 总目标数量统计；
- 当前步骤和总步骤统计；
- 任务冲突拒绝；
- 停止后禁止继续执行。

#### mock_navigation_node

根据任务状态发布模拟导航速度：

```text
导航状态：linear.x = 0.2
其他状态：linear.x = 0.0
```

发布：

```text
/cmd_vel_raw
```

#### task_manager_node

旧版单任务管理器，为兼容早期接口保留。

### 4.5 safety_controller

#### mock_laser_node

发布模拟激光安全数据，用于安全链路测试。

#### safety_filter_node

订阅：

```text
/cmd_vel_raw
```

发布：

```text
/cmd_vel
```

提供服务：

```text
/safety/emergency_stop
```

服务类型：

```text
robot_interfaces/srv/EmergencyStop
```

急停锁定后，安全速度强制为零。人工解除后才允许恢复。

### 4.6 robot_bringup

保存完整系统的 launch 和 YAML 参数。

推荐视觉启动文件：

```text
vision_hybrid_homography_mock.launch.py
```

推荐完整系统启动文件：

```text
full_safe_hybrid_vision_mock.launch.py
```

完整启动文件包含：

- 随机摄像头；
- HSV检测；
- YOLO检测；
- 检测融合；
- 单应性定位；
- Marker可视化；
- DeepSeek任务规划；
- 任务计划管理器；
- 模拟导航；
- 模拟雷达；
- 安全速度过滤。

---

## 5. 训练完成的模型

正式模型：

```text
models/final/red_blue_yolo11n_manual180_960_best.pt
```

模型类别：

```text
0 red_cube
1 blue_cube
```

模型架构：

```text
YOLO11n
```

输入尺寸：

```text
960
```

正式模型已完成训练，不需要合伙人重新训练。

DeepSeek API 不参与视觉模型训练。DeepSeek API 只负责把中文任务转换成机器人任务计划。

已记录的人工测试指标：

```text
Precision: 0.879
Recall: 0.852
mAP50: 0.918
mAP50-95: 0.661
red_cube recall: 0.757
blue_cube recall: 0.947
CPU inference: 约79.2ms
```

最终 HSV+YOLO 无重叠仿真验收：

```text
red_cube: 10/10
blue_cube: 10/10
检测率: 100%
最大边缘误差: 9px
overall_result: PASS
```

详细模型信息见：

```text
docs/TRAINED_MODEL_FACTS.txt
models/final/model_information.txt
```

---

## 6. 新电脑环境要求

推荐环境：

```text
Ubuntu 22.04
ROS 2 Humble
Python 3.10
CPU模式YOLO推理
工作空间路径：~/robot_ws
```

Windows 本身不能直接运行当前 ROS 2 Ubuntu 工作空间。建议在另一台电脑使用 Ubuntu 22.04、双系统或 Ubuntu 虚拟机。

---

## 7. 合伙人克隆仓库后的完整使用方法

### 7.1 克隆到固定工作空间

```bash
git clone <你的私有GitHub仓库地址> ~/robot_ws

cd ~/robot_ws
```

必须克隆到：

```text
~/robot_ws
```

这样模型路径和 YOLO 虚拟环境路径会自动匹配。

### 7.2 安装ROS基础依赖

```bash
sudo apt update

sudo apt install -y \
git \
python3-pip \
python3-venv \
python3-colcon-common-extensions \
python3-rosdep \
ros-humble-cv-bridge \
ros-humble-image-transport \
ros-humble-message-filters \
ros-humble-tf2-ros \
ros-humble-tf2-geometry-msgs \
ros-humble-visualization-msgs \
ros-humble-foxglove-bridge
```

初始化 rosdep：

```bash
sudo rosdep init 2>/dev/null || true

rosdep update

cd ~/robot_ws

rosdep install \
--from-paths src \
--ignore-src \
-r \
-y
```

### 7.3 创建YOLO环境

```bash
cd ~/robot_ws

python3 -m venv \
--system-site-packages \
~/robot_ws/venvs/yolo_cpu

source \
~/robot_ws/venvs/yolo_cpu/bin/activate

python3 -m pip install \
--upgrade pip
```

优先使用仓库保存的版本清单：

```bash
python3 -m pip install \
-r ~/robot_ws/handoff_metadata/yolo_requirements_lock.txt
```

如果完整锁定文件在新系统中存在版本解析问题，可先安装核心依赖：

```bash
python3 -m pip install \
"numpy<2" \
"ultralytics==8.4.120"
```

安装完成后检查：

```bash
python3 -c "
import numpy
import cv2
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
```

不应出现 `_ARRAY_API` 错误。

### 7.4 验证训练模型

```bash
model_file=~/robot_ws/models/final/red_blue_yolo11n_manual180_960_best.pt

test -s "$model_file" \
&& echo "训练模型：正常" \
|| echo "训练模型：缺失"

sha256sum "$model_file"

cat ~/robot_ws/docs/TRAINED_MODEL_FACTS.txt
```

实际 SHA256 必须与 `TRAINED_MODEL_FACTS.txt` 中记录的值一致。

### 7.5 构建ROS工作空间

保持 YOLO 虚拟环境激活，然后执行：

```bash
source /opt/ros/humble/setup.bash

cd ~/robot_ws

colcon build \
--symlink-install

source ~/robot_ws/install/setup.bash
```

检查主要入口：

```bash
ros2 pkg executables vision_detector | sort
ros2 pkg executables deepseek_bridge | sort
ros2 pkg executables task_manager | sort
ros2 pkg executables safety_controller | sort
```

### 7.6 设置DeepSeek API密钥

不要把 API 密钥写进源码、YAML、README 或 GitHub。

在每次启动完整系统前执行：

```bash
read -s -p "请输入DeepSeek API密钥：" DEEPSEEK_API_KEY
echo

export DEEPSEEK_API_KEY
```

检查是否加载：

```bash
if [ -n "$DEEPSEEK_API_KEY" ]; then
    echo "DeepSeek密钥：已加载"
else
    echo "DeepSeek密钥：未加载"
fi
```

合伙人需要使用自己的有效 DeepSeek API 密钥。

更换 API 密钥不会修改、训练或影响本地 YOLO 模型。

### 7.7 启动完整系统

```bash
source ~/robot_ws/venvs/yolo_cpu/bin/activate
source /opt/ros/humble/setup.bash
source ~/robot_ws/install/setup.bash

ros2 launch \
robot_bringup \
full_safe_hybrid_vision_mock.launch.py
```

### 7.8 检查系统

在新终端执行：

```bash
source /opt/ros/humble/setup.bash
source ~/robot_ws/install/setup.bash

ros2 node list | sort

ros2 topic list -t | sort
```

重要话题：

```text
/camera/image_raw
/vision/detections
/vision/yolo_detections
/vision/fused_detections
/vision/localized_detections
/vision/markers
/user/text_command
/llm/task_plan
/task/status
/task/progress
/cmd_vel_raw
/cmd_vel
```

检查定位：

```bash
ros2 topic echo --once \
/vision/localized_detections
```

预期：

```text
frame_id: base_link
has_position: true
```

### 7.9 发送中文任务

```bash
ros2 topic pub --once \
/user/text_command \
std_msgs/msg/String \
"{data: '先把蓝色物块搬到A区一次，再把红色物块搬到C区一次'}"
```

观察状态：

```bash
ros2 topic echo /task/status
```

观察进度：

```bash
ros2 topic echo /task/progress
```

正常任务最终应出现：

```text
state: FINISHED
success: true
```

### 7.10 中文停止

```bash
ros2 topic pub --once \
/user/text_command \
std_msgs/msg/String \
"{data: '停止'}"
```

预期状态：

```text
EMERGENCY_STOP
```

### 7.11 安全急停

锁定：

```bash
ros2 service call \
/safety/emergency_stop \
robot_interfaces/srv/EmergencyStop \
"{engage: true, reason: '人工急停'}"
```

检查：

```bash
ros2 topic echo --once /cmd_vel
```

预期：

```text
linear.x: 0.0
angular.z: 0.0
```

人工解除：

```bash
ros2 service call \
/safety/emergency_stop \
robot_interfaces/srv/EmergencyStop \
"{engage: false, reason: '人工解除'}"
```

---

## 8. 可视化

### RViz

启动：

```bash
rviz2
```

设置：

```text
Fixed Frame: base_link
MarkerArray Topic: /vision/markers
```

### Foxglove

启动：

```bash
ros2 launch \
foxglove_bridge \
foxglove_bridge_launch.xml \
port:=8765
```

连接：

```text
ws://localhost:8765
```

常用话题：

```text
/vision/debug_image
/vision/yolo_debug_image
/vision/markers
/task/status
/task/progress
/cmd_vel
```

---

## 9. 已通过的验收

项目已经完成以下验收：

- ROS消息接口构建；
- DeepSeek单步骤任务解析；
- DeepSeek多步骤任务计划；
- 不完整任务澄清；
- 危险指令拒绝；
- 多步骤顺序执行；
- 任务冲突拒绝；
- 中文停止；
- 安全急停和人工解除；
- HSV红蓝物块检测；
- YOLO模型训练和测试；
- HSV与YOLO检测融合；
- 10次红色检测验收；
- 10次蓝色检测验收；
- 最大边缘误差不超过10像素；
- 五帧位置中值滤波；
- `base_link` 坐标输出；
- MarkerArray可视化；
- 原始速度和安全速度链路；
- 完整中文任务最终 `FINISHED`；
- 急停后原任务不再继续。

验收文件清单见：

```text
docs/ACCEPTANCE_FILE_LIST.txt
```

---

## 10. 当前限制

当前系统包含以下模拟节点：

```text
random_scene_camera_node
mock_navigation_node
mock_laser_node
```

因此当前已经通过的是完整仿真和算法链路验收，还不是实际物理机器人验收。

接入真实机器人时需要：

1. 用真实相机驱动替换随机摄像头；
2. 重新标定真实相机的单应性矩阵；
3. 用真实导航节点替换模拟导航；
4. 用真实雷达驱动替换模拟雷达；
5. 如果使用机械臂，实现真实抓取和放置接口；
6. 在真实环境重新测试红蓝物块检测率；
7. 在真实底盘测试急停和安全速度；
8. 根据真实图像继续补充人工标注数据。

---

## 11. 合伙人建议工作内容

合伙人应从独立分支开始：

```bash
git checkout -b partner/remaining-work
```

建议优先完成：

1. 在新电脑恢复环境并运行最终 launch；
2. 验证训练模型 SHA256；
3. 接入真实摄像头；
4. 标定真实单应性矩阵；
5. 接入真实导航、雷达和机械臂；
6. 保存真实环境验收报告；
7. 推送分支；
8. 创建 Pull Request；
9. 由项目负责人检查并合并。

不要直接向 `main` 分支提交未经测试的代码。

---

## 12. 安全和保密

以下内容禁止提交到 GitHub：

- `DEEPSEEK_API_KEY`
- GitHub访问令牌
- SSH私钥
- `.env`
- 用户密码
- 个人身份信息

以下目录不应提交：

```text
build/
install/
log/
venvs/
__pycache__/
```

这些目录必须在新电脑重新生成。

---

## 13. 推荐启动入口

视觉系统：

```bash
ros2 launch \
robot_bringup \
vision_hybrid_homography_mock.launch.py
```

完整安全系统：

```bash
ros2 launch \
robot_bringup \
full_safe_hybrid_vision_mock.launch.py
```

本项目的正式运行入口是：

```text
full_safe_hybrid_vision_mock.launch.py
```
## 合伙人一键接手方法

### 1. 放置工作空间

请将仓库克隆到固定位置：

```bash
~/robot_ws
