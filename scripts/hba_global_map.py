import os
import numpy as np
import open3d as o3d
import subprocess

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'DATASET'), 'r') as f:
    DATASET_NAME = f.read().strip()
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_ROOT = os.path.join(BASE_DIR, DATASET_NAME)

# HBA 数据目录
HBA_DIR = os.path.join(DATASET_ROOT, 'hba')
HBA_PCD_DIR = os.path.join(HBA_DIR, 'pcd')
HBA_POSE_FILE = os.path.join(HBA_DIR, 'pose.json')

# 输出文件
OUTPUT_MAP_PATH = os.path.join(DATASET_ROOT, 'global_map_hba.pcd')
OUTPUT_MAP_DENSE_PATH = os.path.join(DATASET_ROOT, 'global_map_hba_dense.pcd')

# 可视化配置
VOXEL_SIZE = 0.2  # 地图降采样大小
STATIC_DURATION_THRESH = 1.0   # 判定为静态所需的最小观测时长 (秒)
ENABLE_DYNAMIC_FILTER = True    # 动态物体滤除开关 (True: 开启, False: 直接合并)

def filter_dynamic_obstacles(points_list, filter_voxel_size, time_threshold):
    """
    参考 preprocess_deprecated.py 的动态障碍物滤除逻辑
    """
    print(f"\n[Map Filter] 开始处理动态障碍物滤除...")
    if not points_list:
        return o3d.geometry.PointCloud()
        
    all_points = np.vstack(points_list)
    xyz = all_points[:, :3]
    times = all_points[:, 3]
    
    print(f"  - 总点数: {len(xyz)}")
    print(f"  - 滤波体素大小 (用于检测重合): {filter_voxel_size}m")
    print(f"  - 静态判定时间阈值: {time_threshold}s")
    
    # 1. 体素化
    voxel_indices = np.floor(xyz / filter_voxel_size).astype(np.int64)
    _, inverse_indices = np.unique(voxel_indices, axis=0, return_inverse=True)
    num_voxels = inverse_indices.max() + 1
    
    # 2. 计算每个体素的观测时间跨度
    min_times = np.full(num_voxels, np.inf)
    max_times = np.full(num_voxels, -np.inf)
    np.minimum.at(min_times, inverse_indices, times)
    np.maximum.at(max_times, inverse_indices, times)
    
    durations = max_times - min_times
    
    # 3. 筛选静态体素
    static_voxel_mask = durations >= time_threshold
    valid_point_mask = static_voxel_mask[inverse_indices]
    
    print(f"  - 总子空间(体素)数: {num_voxels}")
    print(f"  - 静态子空间数: {np.sum(static_voxel_mask)} (占比 {np.sum(static_voxel_mask)/num_voxels*100:.1f}%)")
    
    clean_xyz = xyz[valid_point_mask]
    
    filtered_pcd = o3d.geometry.PointCloud()
    filtered_pcd.points = o3d.utility.Vector3dVector(clean_xyz)
    
    keep_ratio = len(clean_xyz) / len(xyz) * 100
    print(f"  - 滤除完成。保留点数: {len(clean_xyz)} ({keep_ratio:.2f}%)")
    return filtered_pcd

def parse_hba_pose(line):
    """
    解析 HBA 位姿行: tx ty tz qw qx qy qz
    返回 4x4 变换矩阵
    """
    data = [float(x) for x in line.split()]
    if len(data) != 7:
        return None
    
    tx, ty, tz, qw, qx, qy, qz = data
    
    # 构建旋转矩阵
    from scipy.spatial.transform import Rotation as R
    rot = R.from_quat([qx, qy, qz, qw]).as_matrix()
    
    T = np.eye(4)
    T[:3, :3] = rot
    T[:3, 3] = [tx, ty, tz]
    return T

def main():
    if not os.path.exists(HBA_PCD_DIR) or not os.path.exists(HBA_POSE_FILE):
        print(f"Error: HBA data not found in {HBA_DIR}")
        return

    print(f"Reading poses from: {HBA_POSE_FILE}")
    with open(HBA_POSE_FILE, 'r') as f:
        pose_lines = f.readlines()

    map_segments_list = []
    
    print(f"Starting to merge {len(pose_lines)} point clouds...")
    
    for i, line in enumerate(pose_lines):
        # 对应文件名 00000.pcd, 00001.pcd...
        pcd_filename = f"{i:05d}.pcd"
        pcd_path = os.path.join(HBA_PCD_DIR, pcd_filename)
        
        if not os.path.exists(pcd_path):
            print(f"\nWarning: {pcd_filename} not found, skipping...")
            continue
            
        T = parse_hba_pose(line)
        if T is None:
            continue
            
        # 读取并变换点云
        pcd = o3d.io.read_point_cloud(pcd_path)
        pcd.transform(T)

        # 模拟时间戳 (参考 generate_global_map.py)
        current_time_s = float(i) * 0.1 
        
        points_np = np.asarray(pcd.points)
        if len(points_np) > 0:
            times_np = np.full((len(points_np), 1), current_time_s)
            segment_data = np.hstack((points_np, times_np))
            map_segments_list.append(segment_data)
        
        print(f"\rCollected: {i+1}/{len(pose_lines)} ({pcd_filename})", end='', flush=True)

    if ENABLE_DYNAMIC_FILTER:
        global_map_pcd = filter_dynamic_obstacles(
            map_segments_list, 
            filter_voxel_size=0.2, 
            time_threshold=STATIC_DURATION_THRESH
        )
    else:
        print("\n[Map Filter] 动态障碍物滤除已关闭，正在直接合并点云...")
        all_points = np.vstack(map_segments_list)[:, :3]
        global_map_pcd = o3d.geometry.PointCloud()
        global_map_pcd.points = o3d.utility.Vector3dVector(all_points)

    print("\nFinalizing global map...")
    
    # 保存稠密地图
    print(f"Saving dense global map to: {OUTPUT_MAP_DENSE_PATH}")
    o3d.io.write_point_cloud(OUTPUT_MAP_DENSE_PATH, global_map_pcd)

    # 降采样并保存
    if VOXEL_SIZE > 0:
        print(f"Downsampling with voxel size: {VOXEL_SIZE}m")
        global_map_pcd = global_map_pcd.voxel_down_sample(VOXEL_SIZE)

    # 保存地图
    print(f"Saving global map to: {OUTPUT_MAP_PATH}")
    o3d.io.write_point_cloud(OUTPUT_MAP_PATH, global_map_pcd)
    print(f"Total points (downsampled): {len(global_map_pcd.points)}")

    # 调用实例 pcl_viewer
    # try:
    #     print(f"Opening pcl_viewer for {OUTPUT_MAP_PATH}...")
    #     subprocess.Popen(['pcl_viewer', OUTPUT_MAP_PATH])
    # except Exception as e:
    #     print(f"Could not open pcl_viewer: {e}")

if __name__ == "__main__":
    main()
