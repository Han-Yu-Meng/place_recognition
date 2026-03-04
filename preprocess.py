# preprocess.py

import os
import sys
import math
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R, Slerp
from collections import deque
import time

# ROS 2 imports
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
import rosbag2_py
import sensor_msgs_py.point_cloud2 as pc2

# ================= 配置参数 =================
DATASET_NAME = 'YunJing'                   # 数据集名称（主文件夹）
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_ROOT = os.path.join(BASE_DIR, DATASET_NAME)

BAG_PATH = os.path.join(DATASET_ROOT, 'bag') # Bag 文件夹路径（现在在子文件夹中）
OUTPUT_DIR = DATASET_ROOT                    # 主输出文件夹
TOPIC_CLOUD = '/cloud_registered_body'
TOPIC_ODOM = '/Odometry'

# --- 关键帧配置 ---
KEYFRAME_DIST = 0.2            # 关键帧距离阈值 (米)
KEYFRAME_ANGLE_DEG = 20.0      # 关键帧角度阈值 (度)
ACCUMULATE_FRAMES = 5          # 关键帧累积帧数

# --- 动态障碍物滤除配置 (仅用于 Global Map 模式) ---
MAP_VOXEL_SIZE = 0.05          # 全局地图处理的体素大小 (米)
STATIC_DURATION_THRESH = 1.5   # 静态物体判定阈值 (秒, 调低一点以适应较短的观测)

LIVOX_FOV_MIN_RAD = np.radians(-7.0)
LIVOX_FOV_MAX_RAD = np.radians(65.0)
INSTALL_PITCH_DEG = 45.0       # 安装角度 (度)

# --- 静态变换 (odom -> lidar_odom) ---
# 用于对 body 系点云进行修正
STATIC_TRANS = np.array([0.5496, 0.2400, 0.1934])
STATIC_QUAT = np.array([0.0, 0.130526, 0.0, 0.991445]) # x, y, z, w

def get_static_tf():
    rot = R.from_quat(STATIC_QUAT).as_matrix()
    T = np.eye(4)
    T[:3, :3] = rot
    T[:3, 3] = STATIC_TRANS
    return T

T_BASE_TO_LIDAR = get_static_tf()

class OdomBuffer:
    def __init__(self, max_size=1000):
        self.buffer = [] # list of (time, pose_matrix)
        self.max_size = max_size
        
    def add(self, time_s, pose_matrix):
        self.buffer.append((time_s, pose_matrix))
        if len(self.buffer) > self.max_size:
            self.buffer.pop(0)
            
    def get_interpolated_pose(self, time_s):
        if not self.buffer:
            return None
        if time_s <= self.buffer[0][0]:
            return self.buffer[0][1]
        if time_s >= self.buffer[-1][0]:
            return self.buffer[-1][1]
            
        for i in range(len(self.buffer) - 1):
            t0, T0 = self.buffer[i]
            t1, T1 = self.buffer[i+1]
            if t0 <= time_s <= t1:
                ratio = (time_s - t0) / (t1 - t0)
                trans = (1 - ratio) * T0[:3, 3] + ratio * T1[:3, 3]
                
                # SLERP 旋转插值
                rots = R.from_matrix([T0[:3, :3], T1[:3, :3]])
                slerp = Slerp([t0, t1], rots)
                interp_rot = slerp([time_s]).as_matrix()[0]
                
                T = np.eye(4)
                T[:3, :3] = interp_rot
                T[:3, 3] = trans
                return T
        return self.buffer[-1][1]

def create_output_dirs(mode):
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    if mode == 'keyframe':
        on_route_dir = os.path.join(OUTPUT_DIR, 'keyframes', 'on_route')
        if not os.path.exists(on_route_dir):
            os.makedirs(on_route_dir)
        return on_route_dir, None
    else:
        return None, OUTPUT_DIR

def get_rosbag_options(path, serialization_format='cdr'):
    storage_options = rosbag2_py.StorageOptions(uri=path, storage_id='sqlite3')
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format=serialization_format,
        output_serialization_format=serialization_format)
    return storage_options, converter_options

def odom_to_matrix(odom_msg):
    p = odom_msg.pose.pose.position
    q = odom_msg.pose.pose.orientation
    
    trans = np.array([p.x, p.y, p.z])
    rot = R.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
    
    T = np.eye(4)
    T[:3, :3] = rot
    T[:3, 3] = trans
    return T

def msg_to_pcd(cloud_msg):
    gen = pc2.read_points(cloud_msg, field_names=("x", "y", "z"), skip_nans=True)
    points = list(gen)
    if not points: return None
    
    raw_array = np.array(points)
    if raw_array.dtype.names:
        xyz = np.column_stack((raw_array['x'], raw_array['y'], raw_array['z']))
    else:
        xyz = raw_array
    xyz = xyz.astype(np.float64)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    return pcd

def calculate_delta(T1, T2):
    T_delta = np.linalg.inv(T1) @ T2
    dist = np.linalg.norm(T_delta[:3, 3])
    rotation = R.from_matrix(T_delta[:3, :3])
    angle_rad = rotation.magnitude()
    angle_deg = np.degrees(angle_rad)
    return dist, angle_deg

def copy_pcd(pcd):
    new_pcd = o3d.geometry.PointCloud()
    new_pcd.points = o3d.utility.Vector3dVector(np.asarray(pcd.points))
    return new_pcd

def accumulate_pcds(frame_list, target_pose):
    """累积点云到 target_pose (Body Frame)"""
    accumulated_feature_pcd = o3d.geometry.PointCloud()
    T_target_inv = np.linalg.inv(target_pose)
    
    for frame in frame_list:
        pcd_temp = copy_pcd(frame['pcd'])
        T_relative = T_target_inv @ frame['pose']
        pcd_temp.transform(T_relative)
        accumulated_feature_pcd += pcd_temp
        
    return accumulated_feature_pcd

def apply_fov_filter(pcd, install_pitch_deg=INSTALL_PITCH_DEG):
    """
    根据安装角度和 FOV 范围裁剪点云。
    pcd: 位于 Body Frame 的点云
    """
    points = np.asarray(pcd.points)
    if len(points) == 0:
        return pcd

    # 1. 构建旋转矩阵 (Body -> Sensor)
    # C++ 代码逻辑: Body_X = Sensor_X * cos(45) + Sensor_Z * sin(45)
    # 这是一个绕 Y 轴旋转 +45 度的过程 (Sensor -> Body)。
    # 因此，从 Body 到 Sensor 需要绕 Y 轴旋转 -45 度。
    theta = np.radians(-install_pitch_deg)
    c, s = np.cos(theta), np.sin(theta)
    
    # 绕 Y 轴旋转矩阵 (Ry)
    # [ c  0  s]
    # [ 0  1  0]
    # [-s  0  c]
    R_body_to_sensor = np.array([
        [c, 0, s],
        [0, 1, 0],
        [-s, 0, c]
    ])
    
    # 变换点云到 Sensor 坐标系
    # points 是 (N, 3)，矩阵乘法需要转置或调整顺序: (R @ P.T).T = P @ R.T
    points_sensor = points @ R_body_to_sensor.T
    
    # 2. 计算每个点在 Sensor 坐标系下的几何属性
    # C++ 模拟器逻辑：点是在球面上生成的，z_sensor (dz) 对应 sin(elevation)
    # 判断标准: sin(fov_min) <= z_sensor / norm <= sin(fov_max)
    
    norms = np.linalg.norm(points_sensor, axis=1)
    
    # 避免除以零
    valid_indices = norms > 1e-6
    
    z_sensor = points_sensor[:, 2] # Z 分量
    
    # 计算 sin(elevation)
    sin_elevation = np.zeros_like(z_sensor)
    sin_elevation[valid_indices] = z_sensor[valid_indices] / norms[valid_indices]
    
    # 3. FOV 过滤
    min_sin = np.sin(LIVOX_FOV_MIN_RAD)
    max_sin = np.sin(LIVOX_FOV_MAX_RAD)
    
    # 保留范围内的点
    mask = (sin_elevation >= min_sin) & (sin_elevation <= max_sin) & valid_indices
    
    filtered_pcd = pcd.select_by_index(np.where(mask)[0])
    
    # print(f"  [FOV Filter] Raw: {len(points)} -> Filtered: {len(filtered_pcd.points)}")
    return filtered_pcd

def filter_dynamic_obstacles(points_list, voxel_size, time_threshold):
    print(f"\n[Map Filter] 开始处理动态障碍物滤除...")
    if not points_list:
        return o3d.geometry.PointCloud()
        
    all_points = np.vstack(points_list)
    xyz = all_points[:, :3]
    times = all_points[:, 3]
    
    print(f"  - 总点数: {len(xyz)}")
    
    # 简单的体素化和时间跨度检查
    voxel_indices = np.floor(xyz / voxel_size).astype(np.int64)
    _, inverse_indices = np.unique(voxel_indices, axis=0, return_inverse=True)
    num_voxels = inverse_indices.max() + 1
    
    min_times = np.full(num_voxels, np.inf)
    max_times = np.full(num_voxels, -np.inf)
    
    np.minimum.at(min_times, inverse_indices, times)
    np.maximum.at(max_times, inverse_indices, times)
    
    durations = max_times - min_times
    static_voxel_mask = durations >= time_threshold
    valid_point_mask = static_voxel_mask[inverse_indices]
    
    clean_xyz = xyz[valid_point_mask]
    
    filtered_pcd = o3d.geometry.PointCloud()
    filtered_pcd.points = o3d.utility.Vector3dVector(clean_xyz)
    
    keep_ratio = len(clean_xyz) / len(xyz) * 100
    print(f"  - 滤除完成。保留点数: {len(clean_xyz)} ({keep_ratio:.2f}%)")
    return filtered_pcd

def main():
    # ================= 用户输入交互 =================
    print("==========================================")
    print("请选择运行模式:")
    print("  [1] 提取关键帧特征 (生成 keyframes/on_route/*.pcd, 应用 FOV 裁剪)")
    print("  [2] 构建全局地图 (生成 global_map.pcd, 去除动态障碍物)")
    print("==========================================")
    
    mode_input = input("请输入数字 [1/2]: ").strip()
    
    if mode_input == '1':
        mode = 'keyframe'
        print(f">> 已选择: 关键帧提取模式 (回访数据来自: {BAG_PATH})")
    elif mode_input == '2':
        mode = 'map'
        print(f">> 已选择: 全局地图构建模式 (回访数据来自: {BAG_PATH})")
    else:
        print("输入无效，退出程序。")
        return

    on_route_dir, _ = create_output_dirs(mode)
        
    storage_options, converter_options = get_rosbag_options(BAG_PATH)
    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)

    topics_types = reader.get_all_topics_and_types()
    type_map = {topic.name: topic.type for topic in topics_types}
    
    metadata = reader.get_metadata()
    try:
        bag_start_time_ns = metadata.starting_time.nanoseconds
    except AttributeError:
        bag_start_time_ns = metadata.starting_time.nanoseconds_since_epoch
    
    last_odom_msg = None
    odom_buffer = OdomBuffer()
    last_keyframe_pose = None
    
    keyframe_buffer = deque(maxlen=ACCUMULATE_FRAMES) 
    keyframe_count = 0
    
    map_segments_list = [] 
    
    print(f"\n开始读取 Bag: {BAG_PATH} ...")

    while reader.has_next():
        (topic, data, t_ns) = reader.read_next()
        current_time_s = (t_ns - bag_start_time_ns) / 1e9
        
        if topic == TOPIC_ODOM:
            msg_type = get_message(type_map[topic])
            last_odom_msg = deserialize_message(data, msg_type)
            odom_buffer.add(current_time_s, odom_to_matrix(last_odom_msg))
            
        elif topic == TOPIC_CLOUD:
            if last_odom_msg is None: continue
            
            # 实时打印进度
            print(f"\r[Processing] Bag Time: {current_time_s:8.2f}s", end='', flush=True)
                
            msg_type = get_message(type_map[topic])
            cloud_msg = deserialize_message(data, msg_type)
            
            current_pcd_lidar = msg_to_pcd(cloud_msg)
            if current_pcd_lidar is None: continue

            T_curr = odom_buffer.get_interpolated_pose(current_time_s)
            if T_curr is None: T_curr = odom_to_matrix(last_odom_msg)
            
            # ---------------------------------------------------------
            # 模式 2: 全局地图数据收集
            # ---------------------------------------------------------
            if mode == 'map':
                pcd_world = copy_pcd(current_pcd_lidar)
                pcd_world.transform(T_curr)
                pcd_world.transform(T_BASE_TO_LIDAR)
                pcd_world = pcd_world.voxel_down_sample(voxel_size=0.05) 
                
                points_np = np.asarray(pcd_world.points)
                if len(points_np) > 0:
                    times_np = np.full((len(points_np), 1), current_time_s)
                    segment_data = np.hstack((points_np, times_np))
                    map_segments_list.append(segment_data)

            # ---------------------------------------------------------
            # 模式 1 & 2: 关键帧判断逻辑 (模式2仅用于推进，不保存)
            # ---------------------------------------------------------
            current_frame_data = {
                'pcd': current_pcd_lidar, # 这里的 PCD 是 Body Frame
                'pose': T_curr,
                'time_s': current_time_s
            }
            keyframe_buffer.append(current_frame_data)
            
            is_keyframe = False
            if last_keyframe_pose is None:
                is_keyframe = True
            else:
                dist, angle = calculate_delta(last_keyframe_pose, T_curr)
                if dist >= KEYFRAME_DIST or angle >= KEYFRAME_ANGLE_DEG:
                    is_keyframe = True
            
            if is_keyframe:
                if len(keyframe_buffer) == ACCUMULATE_FRAMES:
                    # 仅在模式 1 下处理和保存关键帧
                    if mode == 'keyframe':
                        # 1. 累积点云 (结果在当前 Keyframe 的 Body Frame)
                        accumulated_pcd = accumulate_pcds(keyframe_buffer, T_curr)
                        
                        # 2. 应用外部修正 (Body -> Lidar_Odom)
                        # 为了让关键帧的点云与全局地图对齐，需要应用 T_BASE_TO_LIDAR 变换
                        accumulated_pcd.transform(T_BASE_TO_LIDAR)
                        
                        # 3. 应用 FOV 裁剪 (模拟 Sensor 视野)
                        # accumulated_pcd = apply_fov_filter(accumulated_pcd)

                        # 4. 降采样并保存
                        feature_filename = os.path.join(on_route_dir, f"{keyframe_count:06d}.pcd")
                        pose_filename = os.path.join(on_route_dir, f"{keyframe_count:06d}.odom")
                        
                        accumulated_pcd = accumulated_pcd.voxel_down_sample(voxel_size=0.1)
                        o3d.io.write_point_cloud(feature_filename, accumulated_pcd)
                        
                        # 保存也需要保存变换后的 Pose
                        T_ext = T_BASE_TO_LIDAR
                        T_ext_inv = np.linalg.inv(T_ext)
                        T_body_in_map = T_ext @ T_curr @ T_ext_inv

                        np.savetxt(pose_filename, T_body_in_map)
                        print(f"\n[KeyFrame] Saved {keyframe_count:06d} (Points: {len(accumulated_pcd.points)})")

                    last_keyframe_pose = T_curr
                    keyframe_count += 1

    # ================= 处理结束后的工作 =================
    print("\nProcessing finished.")
    
    if mode == 'map':
        print("\n--- Building Global Map with Dynamic Obstacle Removal ---")
        global_map_pcd = filter_dynamic_obstacles(
            map_segments_list, 
            voxel_size=MAP_VOXEL_SIZE, 
            time_threshold=STATIC_DURATION_THRESH
        )

        print("\n正在保存最终全局地图...")
        final_map_pcd = global_map_pcd.voxel_down_sample(voxel_size=0.05)
        o3d.io.write_point_cloud(os.path.join(OUTPUT_DIR, "global_map.pcd"), final_map_pcd)
        print(f"处理完成！全局地图已保存至 {OUTPUT_DIR}/global_map.pcd")
    
    elif mode == 'keyframe':
        print(f"\n关键帧提取完成。共生成 {keyframe_count} 帧。")

if __name__ == "__main__":
    main()