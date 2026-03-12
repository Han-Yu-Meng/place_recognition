# 项目介绍

本项目用于激光雷达地点识别（Place Recognition）的前处理，包括从 ROS Bag 提取关键帧、利用 `interactive_slam` 进行轨迹优化、构建全局地图以及基于全局地图模拟生成多视角/多配置的仿真点云。

## 项目结构与文件说明

### 核心代码 (C++)
- [include/lidar_simulator.hpp](include/lidar_simulator.hpp): 核心雷达仿真器，支持射线投影 (Ray Casting)、隐点剔除 (HPR) 和 Z-Buffer 深度图模拟等多种仿真策略。
- [include/sc_module.hpp](include/sc_module.hpp): Scan Context 管理器，用于生成、存储、检索雷达描述子，并提供多视图调试可视化功能。
- [src/keyframe_simulator.cpp](src/keyframe_simulator.cpp): 点云仿真工具，根据位姿文件在全局地图中批量生成仿真点云帧。
- [src/match_test.cpp](src/match_test.cpp): 匹配测试工具，对比真实采集的关键帧与仿真生成的关键帧，验证地点识别的准确率。
- [src/simulator_test.cpp](src/simulator_test.cpp): 仿真器测试工具，用于生成真实点云与仿真点云的对比 PCD（带颜色区分）。

### 脚本工具 (Python)
- [scripts/process_bag.py](scripts/process_bag.py): 从 `.db3` 格式的 ROS Bag 中根据距离/角度阈值提取关键帧点云及位姿。
- [scripts/graph2odometry.py](scripts/graph2odometry.py): 将 `interactive_slam` 优化后的位姿及点云转换回标准 PCD + Odometry 格式。
- [scripts/generate_global_map.py](scripts/generate_global_map.py): 融合所有优化后的关键帧以生成全局地图，支持体素降采样和动态障碍物滤除。

## 使用流程

数据文件夹的结构：


YunJing/
├── bag/                # 存放 rosbag 文件
├── interactive_slam/
├───── raw/             # 修正前的点云帧
├───── processed/       # 修正后的点云帧
├── keyframes/
├───── raw/             # 修正前的点云帧
├───── processed/       # 修正后的点云帧
├───── simulated/       # 模拟点云帧
├─── global_map.pcd     # 去掉动态障碍物后的全局地图
├─── lidar_config.yaml  # LiDAR 配置文件


所有的 .py 文件都在 scripts/ 文件夹中，所有的 .cpp 文件都在 src/ 文件夹中。

1. 运行 process_bag.py，将 rosbag 中的数据转换为多个 .odom + .pcd 文件的 ROS 数据格式，保存在 keyframes/raw 文件夹中。

   - 注意确认 base_link->lidar_frame 的变换是否正确

2. 在 interactive_slam 的 docker 环境中，运行 rosrun interactive_slam odometry2graph，将 keyframes/raw 中的 ROS 数据格式转换为 interactive_slam 专有格式，保存在 interactive_slam/raw 文件夹中。

   - 使用默认配置，间隔 3m 记录关键帧。

3. 在 interactive_slam 的 docker 环境中，运行 rosrun interactive_slam interactive_slam，使用 interactive_slam 进行点云修正，生成修正后的点云数据，保存在 interactive_slam/processed 文件夹中。

   - 确保修正后，点云无重影，地面基本平坦。

4. 运行 graph2odometry.py，将 interactive_slam/processed 中的点云数据转换回 ROS 数据格式，保存在 keyframes/processed 文件夹中。

5. 运行 generate_global_map.py，将 keyframes/processed 中的点云数据进行融合，生成去掉动态障碍物后的全局地图 global_map.pcd。

6. 关键帧位点选择，以及模拟点云帧生成，运行 C++ 程序 keyframe_simulator.cpp，交互式生成模拟点云帧，保存在 keyframes/simulated 文件夹中。