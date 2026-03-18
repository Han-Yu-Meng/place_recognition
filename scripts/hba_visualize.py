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

# 可视化配置
VOXEL_SIZE = 0.2  # 地图降采样大小

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

    global_pcd = o3d.geometry.PointCloud()
    
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
        
        global_pcd += pcd
        
        print(f"\rMerged: {i+1}/{len(pose_lines)} ({pcd_filename})", end='', flush=True)

    print("\n\nFinalizing global map...")
    if VOXEL_SIZE > 0:
        print(f"Downsampling with voxel size: {VOXEL_SIZE}m")
        global_pcd = global_pcd.voxel_down_sample(VOXEL_SIZE)

    # 保存地图
    print(f"Saving global map to: {OUTPUT_MAP_PATH}")
    o3d.io.write_point_cloud(OUTPUT_MAP_PATH, global_pcd)
    print(f"Total points: {len(global_pcd.points)}")

    # 调用实例 pcl_viewer
    try:
        print(f"Opening pcl_viewer for {OUTPUT_MAP_PATH}...")
        subprocess.Popen(['pcl_viewer', OUTPUT_MAP_PATH])
    except Exception as e:
        print(f"Could not open pcl_viewer: {e}")

if __name__ == "__main__":
    main()
