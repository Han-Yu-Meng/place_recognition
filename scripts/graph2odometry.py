import os
import numpy as np
import open3d as o3d

# ================= 配置参数 =================
DATASET_NAME = 'YunJing'
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_ROOT = os.path.join(BASE_DIR, DATASET_NAME)

# 输入输出目录
RAW_INPUT_DIR = os.path.join(DATASET_ROOT, 'interactive_slam', 'raw')
PROCESSED_INPUT_DIR = os.path.join(DATASET_ROOT, 'interactive_slam', 'processed')
OUTPUT_DIR = os.path.join(DATASET_ROOT, 'keyframes', 'processed')

def parse_data_file(file_path):
    """
    解析 data 文件，提取 estimate 和 odom 矩阵
    """
    data_info = {'estimate': None, 'odom': None}
    
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                if line.startswith('estimate'):
                    matrix = []
                    for j in range(1, 5):
                        matrix.append([float(x) for x in lines[i+j].split()])
                    data_info['estimate'] = np.array(matrix)
                elif line.startswith('odom'):
                    matrix = []
                    for j in range(1, 5):
                        matrix.append([float(x) for x in lines[i+j].split()])
                    data_info['odom'] = np.array(matrix)
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        return None
        
    return data_info

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        print(f"Created output directory: {OUTPUT_DIR}\n")

    # 基于 processed 文件夹中的子目录(ID)进行遍历
    subdirs = sorted([d for d in os.listdir(PROCESSED_INPUT_DIR) if os.path.isdir(os.path.join(PROCESSED_INPUT_DIR, d))])
    
    if not subdirs:
        print(f"No subdirectories found in {PROCESSED_INPUT_DIR}")
        return

    processed_count = 0
    print(f"Starting precise conversion based on Odometry equivalence...")

    for subdir in subdirs:
        processed_data_file = os.path.join(PROCESSED_INPUT_DIR, subdir, 'data')
        raw_data_file = os.path.join(RAW_INPUT_DIR, subdir, 'data')
        raw_pcd_path = os.path.join(RAW_INPUT_DIR, subdir, 'raw.pcd')
        
        # 确保对应文件存在
        if not (os.path.exists(processed_data_file) and os.path.exists(raw_data_file) and os.path.exists(raw_pcd_path)):
            continue
            
        # 1. 解析 raw 和 processed 的 data 文件
        raw_data = parse_data_file(raw_data_file)
        proc_data = parse_data_file(processed_data_file)
        
        if raw_data['odom'] is None or proc_data['odom'] is None or proc_data['estimate'] is None:
            print(f"\nWarning: Missing matrix data in ID {subdir}, skipping.")
            continue
            
        T_raw_odom = raw_data['odom']
        T_proc_odom = proc_data['odom']
        T_proc_estimate = proc_data['estimate']
        
        # 2. 核心等式：计算坐标系对齐矩阵
        # 根据推论：T_raw_odom * P_raw = T_proc_odom * P_cloud
        # 得出：P_cloud = inv(T_proc_odom) * T_raw_odom * P_raw
        T_align = np.linalg.inv(T_proc_odom) @ T_raw_odom
        
        # 3. 读取原始稠密点云，并施加变换
        raw_pcd = o3d.io.read_point_cloud(raw_pcd_path)
        raw_pcd.transform(T_align)
        
        # 4. 保存结果到 ROS 格式目录
        output_pcd_path = os.path.join(OUTPUT_DIR, f"{subdir}.pcd")
        output_odom_path = os.path.join(OUTPUT_DIR, f"{subdir}.odom")
        
        # 保存变换后的点云 (现在的点云局部坐标系和 cloud.pcd 已经绝对一致)
        o3d.io.write_point_cloud(output_pcd_path, raw_pcd)
        # 对应的全局位姿直接使用优化后的 estimate
        np.savetxt(output_odom_path, T_proc_estimate, fmt='%.10f')
        
        processed_count += 1
        print(f"\rProcessed: ID {subdir} ({processed_count}/{len(subdirs)})", end='', flush=True)

    print(f"\n\nFinished! Total converted: {processed_count} keyframes.")
    print(f"All dense pointclouds and optimized odometry are ready in: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()