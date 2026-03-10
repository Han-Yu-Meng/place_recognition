# 使用流程

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