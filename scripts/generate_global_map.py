import os
import numpy as np
import open3d as o3d
import glob
import subprocess

# ================= 配置参数 =================
DATASET_NAME = 'YunJing'
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_ROOT = os.path.join(BASE_DIR, DATASET_NAME)

# 输入目录：经过 interactive_slam 修正并转换回 ROS 格式的关键帧
INPUT_DIR = os.path.join(DATASET_ROOT, 'keyframes', 'processed')
# 输出文件：全局地图
OUTPUT_MAP_PATH = os.path.join(DATASET_ROOT, 'global_map.pcd')

# --- 地图生成配置 ---
MAP_VOXEL_SIZE = 0.2          # 全局地图体素降采样大小 (米)
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

def main():
    if not os.path.exists(INPUT_DIR):
        print(f"Error: Input directory {INPUT_DIR} does not exist.")
        return

    pcd_files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.pcd")))
    if not pcd_files:
        print(f"No .pcd files found in {INPUT_DIR}")
        return

    print(f"Starting to generate global map from {len(pcd_files)} keyframes...")
    
    map_segments_list = []
    processed_count = 0

    for i, pcd_path in enumerate(pcd_files):
        odom_path = pcd_path.replace(".pcd", ".odom")
        if not os.path.exists(odom_path):
            continue
            
        pcd = o3d.io.read_point_cloud(pcd_path)
        try:
            pose = np.loadtxt(odom_path)
        except:
            continue
        pcd.transform(pose)
        
        # 修正：将模拟关键帧间隔从 0.5s 减小到 0.1s
        # 128 帧对应约 13 秒的数据，更有利于 threshold 命中
        current_time_s = float(i) * 0.1 
        
        points_np = np.asarray(pcd.points)
        if len(points_np) > 0:
            times_np = np.full((len(points_np), 1), current_time_s)
            segment_data = np.hstack((points_np, times_np))
            map_segments_list.append(segment_data)
            
        processed_count += 1
        print(f"\rCollected: {processed_count}/{len(pcd_files)}", end='', flush=True)

    if ENABLE_DYNAMIC_FILTER:
        # 动态障碍物滤除核心调整：
        # 将用于检测重合的 filter_voxel_size 从 0.05 调大到 0.2
        # 这样即使定位有微小漂移，同一个静态地物也能落在同一个体素内，从而满足时间阈值。
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

    print("\nFinalizing and saving global map...")
    final_pcd = global_map_pcd.voxel_down_sample(voxel_size=MAP_VOXEL_SIZE)
    o3d.io.write_point_cloud(OUTPUT_MAP_PATH, final_pcd)
    
    print(f"Global map saved to: {OUTPUT_MAP_PATH}")
    print(f"Total points: {len(final_pcd.points)}")

    try:
        subprocess.Popen(['pcl_viewer', OUTPUT_MAP_PATH])
    except:
        pass

if __name__ == "__main__":
    main()
