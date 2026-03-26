import os
import glob
import shutil
import numpy as np
from scipy.spatial.transform import Rotation as R

# 读取当前数据集名称
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'DATASET'), 'r') as f:
    DATASET_NAME = f.read().strip()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_ROOT = os.path.join(BASE_DIR, DATASET_NAME)

# 输入输出路径
INPUT_DIR = os.path.join(DATASET_ROOT, 'keyframes', 'processed')
HBA_DIR = os.path.join(DATASET_ROOT, 'hba')
HBA_PCD_DIR = os.path.join(HBA_DIR, 'pcd')
HBA_POSE_FILE = os.path.join(HBA_DIR, 'pose.json')

def load_odom(file_path):
    """读取 4x4 变换矩阵"""
    return np.loadtxt(file_path)

def main():
    if not os.path.exists(HBA_PCD_DIR):
        os.makedirs(HBA_PCD_DIR, exist_ok=True)
        print(f"Created HBA PCD directory: {HBA_PCD_DIR}")

    # 获取所有 processed odom 文件
    odom_files = sorted(glob.glob(os.path.join(INPUT_DIR, '*.odom')))
    if not odom_files:
        print(f"No odom files found in {INPUT_DIR}")
        return

    hba_poses = []
    print(f"Processing {len(odom_files)} frames from {INPUT_DIR} to HBA format...")

    for i, odom_path in enumerate(odom_files):
        # 获取文件名 (不含扩展名)
        basename = os.path.basename(odom_path).replace('.odom', '')
        
        # 1. 处理位姿
        T = load_odom(odom_path)
        pos = T[:3, 3]
        quat = R.from_matrix(T[:3, :3]).as_quat() # [x, y, z, w]
        
        # HBA 格式: tx ty tz qw qx qy qz
        hba_pose_str = f"{pos[0]} {pos[1]} {pos[2]} {quat[3]} {quat[0]} {quat[1]} {quat[2]}"
        hba_poses.append(hba_pose_str)

        # 2. 拷贝并重命名 PCD 文件 (使用 5 位数补全，如 00000.pcd)
        src_pcd = os.path.join(INPUT_DIR, f"{basename}.pcd")
        dst_pcd = os.path.join(HBA_PCD_DIR, f"{i:05d}.pcd")
        
        if os.path.exists(src_pcd):
            shutil.copy(src_pcd, dst_pcd)
        else:
            print(f"Warning: PCD file not found for {basename}: {src_pcd}")

        if i % 100 == 0 or i == len(odom_files) - 1:
            print(f"\rProcessed {i+1}/{len(odom_files)}", end='', flush=True)

    # 保存 pose.json (HBA 实际上期望的是文本格式)
    with open(HBA_POSE_FILE, 'w') as f:
        f.write("\n".join(hba_poses))

    print(f"\n\nFinished! Total keyframes: {len(hba_poses)}")
    print(f"HBA data saved in: {HBA_DIR}")

if __name__ == "__main__":
    main()
