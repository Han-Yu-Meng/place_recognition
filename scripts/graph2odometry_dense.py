import os
import glob
import shutil
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

# ================= 配置参数 =================
DATASET_NAME = 'YunJing'
# 自动定位到项目根目录
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_ROOT = os.path.join(BASE_DIR, DATASET_NAME)

# 输入输出目录
RAW_KF_DIR = os.path.join(DATASET_ROOT, 'keyframes', 'raw')
PROCESSED_INPUT_DIR = os.path.join(DATASET_ROOT, 'interactive_slam', 'processed')
OUTPUT_DIR = os.path.join(DATASET_ROOT, 'keyframes', 'processed')

def parse_data_file(file_path):
    """
    解析 data 文件，提取 stamp_id (对应原始帧), estimate 和 odom 矩阵
    """
    data_info = {'stamp_id': None, 'estimate': None, 'odom': None}
    
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                if line.startswith('stamp'):
                    # 解析如: stamp 555 0 -> 提取 555
                    parts = line.split()
                    if len(parts) >= 2:
                        data_info['stamp_id'] = int(parts[1])
                elif line.startswith('estimate'):
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

def load_odom(file_path):
    """读取 4x4 变换矩阵"""
    return np.loadtxt(file_path)

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        print(f"Created output directory: {OUTPUT_DIR}\n")

    # ================= 1. 获取所有原始帧的 ID =================
    raw_odom_files = glob.glob(os.path.join(RAW_KF_DIR, '*.odom'))
    raw_ids = []
    for f in raw_odom_files:
        basename = os.path.basename(f)
        frame_id = int(basename.split('.')[0])
        raw_ids.append(frame_id)
    raw_ids.sort()

    if not raw_ids:
        print(f"No raw odometry files found in {RAW_KF_DIR}")
        return
    print(f"Found {len(raw_ids)} dense raw frames in {RAW_KF_DIR}.")

    # ================= 2. 读取优化后的关键帧 =================
    subdirs = sorted([d for d in os.listdir(PROCESSED_INPUT_DIR) if os.path.isdir(os.path.join(PROCESSED_INPUT_DIR, d))])
    
    kf_data_list = []
    seen_ids = set()
    for subdir in subdirs:
        data_file = os.path.join(PROCESSED_INPUT_DIR, subdir, 'data')
        if not os.path.exists(data_file):
            continue
            
        data = parse_data_file(data_file)
        if data and data['stamp_id'] is not None and data['estimate'] is not None:
            stamp_id = data['stamp_id']
            # 防止重复帧导致 Slerp 插值崩溃
            if stamp_id not in seen_ids:
                seen_ids.add(stamp_id)
                kf_data_list.append(data)
    
    if len(kf_data_list) < 2:
        print("Error: Need at least 2 valid optimized keyframes for interpolation.")
        return

    # 按照原始帧 ID 从小到大排序
    kf_data_list.sort(key=lambda x: x['stamp_id'])
    print(f"Parsed {len(kf_data_list)} optimized sparse keyframes.")

    # ================= 3. 计算关键帧上的误差漂移 (Drift) =================
    kf_ids = []
    delta_translations = []
    delta_quats = []

    print("Computing SE(3) drift corrections at keyframes...")
    for data in kf_data_list:
        stamp_id = data['stamp_id']
        stamp_str = f"{stamp_id:06d}"
        T_opt = data['estimate']
        
        raw_odom_path = os.path.join(RAW_KF_DIR, f"{stamp_str}.odom")
        if not os.path.exists(raw_odom_path):
            print(f"Warning: Raw odometry file not found for stamp_id {stamp_id} ({stamp_str}): {raw_odom_path}")
            continue
            
        T_raw = load_odom(raw_odom_path)
        
        # 核心逻辑：计算误差变换矩阵 Delta_T
        # 满足关系: Delta_T * T_raw = T_opt  =>  Delta_T = T_opt * inv(T_raw)
        T_raw_inv = np.linalg.inv(T_raw)
        Delta_T = T_opt @ T_raw_inv
        
        kf_ids.append(stamp_id)
        delta_translations.append(Delta_T[:3, 3])
        # 提取旋转矩阵并转为四元数
        r = R.from_matrix(Delta_T[:3, :3])
        delta_quats.append(r.as_quat())

    kf_ids = np.array(kf_ids)
    delta_translations = np.array(delta_translations)
    
    if len(kf_ids) == 0:
        print("Error: No valid keyframes found with existing raw odometry files. Cannot continue.")
        return

    # 构建球面线性插值器 (Slerp)
    rotations = R.from_quat(delta_quats)
    slerp = Slerp(kf_ids, rotations)

    # ================= 4. 对所有稠密帧插值误差并应用 =================
    print("Interpolating drift and generating dense processed keyframes...")
    processed_count = 0
    
    for i in raw_ids:
        i_str = f"{i:06d}"
        # 处理边界情况 (早于第一个关键帧或晚于最后一个关键帧，维持常量误差)
        if i <= kf_ids[0]:
            t = delta_translations[0]
            r = rotations[0]
        elif i >= kf_ids[-1]:
            t = delta_translations[-1]
            r = rotations[-1]
        else:
            # 找到相邻的关键帧进行插值
            idx_right = np.searchsorted(kf_ids, i)
            idx_left = idx_right - 1
            
            id_L = kf_ids[idx_left]
            id_R = kf_ids[idx_right]
            
            # 线性插值比例
            alpha = (i - id_L) / float(id_R - id_L)
            
            # 平移部分线性插值
            t_L = delta_translations[idx_left]
            t_R = delta_translations[idx_right]
            t = (1.0 - alpha) * t_L + alpha * t_R
            
            # 旋转部分球面插值
            r = slerp(i)
            
        # 重构当前帧的误差修正矩阵 Delta_T
        Delta_T = np.eye(4)
        Delta_T[:3, :3] = r.as_matrix()
        Delta_T[:3, 3] = t
        
        # 读取当前帧原始位姿
        raw_odom_path = os.path.join(RAW_KF_DIR, f"{i_str}.odom")
        T_raw = load_odom(raw_odom_path)
        
        # 施加修正得到最终位姿
        T_final = Delta_T @ T_raw
        
        # 保存新的全局位姿
        np.savetxt(os.path.join(OUTPUT_DIR, f"{i_str}.odom"), T_final, fmt='%.10f')
        
        # ================= 关键优化 =================
        # 由于原始点云是在本地坐标系(Body/LiDAR)下，点坐标本身不需要做任何变换。
        # 我们只更新了 odom，所以直接拷贝 PCD 文件即可，这极大地提高了速度！
        raw_pcd_path = os.path.join(RAW_KF_DIR, f"{i_str}.pcd")
        out_pcd_path = os.path.join(OUTPUT_DIR, f"{i_str}.pcd")
        if os.path.exists(raw_pcd_path):
            shutil.copy(raw_pcd_path, out_pcd_path)
        
        processed_count += 1
        print(f"\rProcessed frame {i_str} ({processed_count}/{len(raw_ids)})", end='', flush=True)
        
    print(f"\n\nFinished! Successfully generated {processed_count} dense trajectory frames.")
    print(f"All optimized dense outputs are ready in: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()