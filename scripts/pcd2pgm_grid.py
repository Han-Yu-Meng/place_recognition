import os
import numpy as np
import open3d as o3d
from PIL import Image
import yaml
import cv2  # 用于图像形态学处理，连结断开的墙壁

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'DATASET'), 'r') as f:
    DATASET_NAME = f.read().strip()
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_ROOT = os.path.join(BASE_DIR, DATASET_NAME)

# 输入/输出文件路径
INPUT_MAP_PATH = os.path.join(DATASET_ROOT, 'global_map_dense.pcd')
OUTPUT_PGM_PATH = os.path.join(DATASET_ROOT, 'global_map.pgm')
OUTPUT_PNG_PATH = os.path.join(DATASET_ROOT, 'global_map.png')
OUTPUT_GROUND_PNG_PATH = os.path.join(DATASET_ROOT, 'ground_height.png')
OUTPUT_YAML_PATH = os.path.join(DATASET_ROOT, 'global_map.yaml')

MAP_RESOLUTION = 0.05      # 地图分辨率 (米/像素)

# --- 1. 绝对高度过滤 (一刀切掉天空和地下) ---
ABS_Z_MIN = -1.0           # 绝对最小高度 (过滤地下噪点/多径反射)
ABS_Z_MAX = 3.0            # 绝对最大高度 (直接切掉高于3米的屋檐、天花板、树冠)

# --- 2. 相对高度与地面校验 (核心逻辑) ---
MIN_OBS_HEIGHT = 0.3       # 障碍物最小相对高度(米)，高于局部地面0.3米才算障碍物
MAX_OBS_HEIGHT = 2.5       # 障碍物最大相对高度(米)，低于局部地面2.0米
MAX_ELEVATION_CHANGE = 2.0 # ★ 允许的最大地形起伏差。如果局部地面比全局地面高出 1.5m，则认为它是悬空物(如绳索)，而非地面！

# --- 3. 噪点滤波配置 ---
THRE_RADIUS = 0.5          # 半径滤波器半径
THRES_POINT_COUNT = 5      # 半径内最小点数

ODOM_TO_LIDAR_ODOM = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

def apply_transform(pcd, transform_params):
    if all(v == 0.0 for v in transform_params):
        return pcd
    return pcd

def extract_robust_obstacles(pcd, resolution):
    """
    鲁棒的障碍物提取：结合绝对高度截断与全局基准面校验
    """
    points = np.asarray(pcd.points)
    print(f"Original points: {len(points)}")
    
    # ---------------- 步骤 1：绝对高度预过滤 ----------------
    # 直接砍掉过高(屋顶/天空)和过低(地下噪点)的点
    valid_z_mask = (points[:, 2] >= ABS_Z_MIN) & (points[:, 2] <= ABS_Z_MAX)
    points = points[valid_z_mask]
    print(f"After Absolute Z Cut ({ABS_Z_MIN}m to {ABS_Z_MAX}m): {len(points)} points")

    if len(points) == 0:
        return None, 0, 0, 0, 0

    # ---------------- 步骤 2：计算网格边界与索引 ----------------
    min_x, min_y = np.min(points[:, :2], axis=0)
    max_x, max_y = np.max(points[:, :2], axis=0)
    
    width = int(np.ceil((max_x - min_x) / resolution))
    height = int(np.ceil((max_y - min_y) / resolution))
    
    ix = np.clip(np.floor((points[:, 0] - min_x) / resolution).astype(int), 0, width - 1)
    iy = np.clip(np.floor((points[:, 1] - min_y) / resolution).astype(int), 0, height - 1)
    
    # ---------------- 步骤 3：计算局部网格最低点 ----------------
    min_z_grid = np.full((height, width), np.inf)
    np.minimum.at(min_z_grid, (iy, ix), points[:, 2])
    
    # ---------------- 步骤 4：★ 全局基准面校验 (解决绳索/屋檐问题) ★ ----------------
    # 使用第 5% 分位数的 Z 值作为"全局基准地面" (比直接用最小值更鲁棒，排除了极低噪点)
    global_base_z = np.percentile(points[:, 2], 5)
    print(f"Estimated Global Base Ground Z: {global_base_z:.2f} m")
    
    # 找出那些"假地面" (最低点比全局基准面高出太多的网格，说明里面悬空了，只有绳索/屋檐)
    invalid_ground_mask = min_z_grid > (global_base_z + MAX_ELEVATION_CHANGE)
    
    # 将假地面的高度设为正无穷，这样它们里面就不会提取出任何障碍物
    min_z_grid[invalid_ground_mask] = np.inf
    
    # ---------------- 步骤 5：提取相对高度范围内的障碍物 ----------------
    local_ground_z = min_z_grid[iy, ix] # 获取每个点对应的(已校验过的)局部地面高度
    
    # 只有当点高于有效局部地面 0.3m，且低于 2.0m 时，才认为是障碍物
    obs_mask = (points[:, 2] >= local_ground_z + MIN_OBS_HEIGHT) & \
               (points[:, 2] <= local_ground_z + MAX_OBS_HEIGHT)
               
    obstacle_points = points[obs_mask]
    
    obs_pcd = o3d.geometry.PointCloud()
    obs_pcd.points = o3d.utility.Vector3dVector(obstacle_points)
    
    return obs_pcd, min_x, min_y, width, height, min_z_grid


def save_ground_height_heatmap(min_z_grid, output_path):
    """
    将地面高度网格转换为热力图并保存
    """
    print(f"Saving ground height heatmap to {output_path}...")
    valid_mask = np.isfinite(min_z_grid)
    if not np.any(valid_mask):
        print("Warning: No valid ground points for heatmap.")
        return

    # 归一化并生成伪彩色图
    min_val = np.min(min_z_grid[valid_mask])
    max_val = np.max(min_z_grid[valid_mask])
    
    # 转换为 0-255
    normalized = np.zeros_like(min_z_grid, dtype=np.uint8)
    normalized[valid_mask] = ((min_z_grid[valid_mask] - min_val) / (max_val - min_val + 1e-6) * 255).astype(np.uint8)
    
    # 翻转 Y 轴以匹配图像坐标系
    normalized_flipped = np.flipud(normalized)
    valid_mask_flipped = np.flipud(valid_mask)
    
    color_map = cv2.applyColorMap(normalized_flipped, cv2.COLORMAP_JET)
    color_map[~valid_mask_flipped] = 0 # 无效区域设为黑
    
    cv2.imwrite(output_path, color_map)


def main():
    if not os.path.exists(INPUT_MAP_PATH):
        print(f"Error: Input map {INPUT_MAP_PATH} does not exist.")
        return

    print(f"Loading global map from {INPUT_MAP_PATH}...")
    pcd = o3d.io.read_point_cloud(INPUT_MAP_PATH)

    pcd = apply_transform(pcd, ODOM_TO_LIDAR_ODOM)

    # 1. 提取障碍物 (使用改进后的鲁棒算法)
    obs_pcd, x_min, y_min, width, height, min_z_grid = extract_robust_obstacles(pcd, MAP_RESOLUTION)
    
    if obs_pcd is None or len(obs_pcd.points) == 0:
        print("Error: Point cloud is empty after obstacle extraction!")
        return
        
    print(f"Obstacles extracted: {len(obs_pcd.points)} points")

    # 1.5 保存地面高度热力图
    save_ground_height_heatmap(min_z_grid, OUTPUT_GROUND_PNG_PATH)

    # 2. 半径去噪
    print(f"Applying RadiusOutlier filter...")
    obs_pcd, _ = obs_pcd.remove_radius_outlier(nb_points=THRES_POINT_COUNT, radius=THRE_RADIUS)
    print(f"After RadiusOutlier: {len(obs_pcd.points)} points")

    # 3. 投影到 2D 栅格
    obs_points = np.asarray(obs_pcd.points)
    map_data = np.full((height, width), 255, dtype=np.uint8)

    for pt in obs_points:
        ix = int(np.floor((pt[0] - x_min) / MAP_RESOLUTION))
        iy = int(np.floor((pt[1] - y_min) / MAP_RESOLUTION))
        if 0 <= ix < width and 0 <= iy < height:
            map_data[height - 1 - iy, ix] = 0

    # 4. 形态学闭运算 (连接断断续续的墙壁)
    print("Applying Morphological Closing to connect disconnected walls...")
    binary_map = 255 - map_data
    kernel = np.ones((3, 3), np.uint8)
    processed_binary_map = cv2.morphologyEx(binary_map, cv2.MORPH_CLOSE, kernel, iterations=2)
    final_map_data = 255 - processed_binary_map

    # 5. 保存文件
    img = Image.fromarray(final_map_data)
    img.save(OUTPUT_PGM_PATH)
    img.save(OUTPUT_PNG_PATH)

    yaml_data = {
        'image': os.path.basename(OUTPUT_PGM_PATH),
        'resolution': MAP_RESOLUTION,
        'origin': [float(x_min), float(y_min), 0.0],
        'negate': 0,
        'occupied_thresh': 0.65,
        'free_thresh': 0.196
    }

    with open(OUTPUT_YAML_PATH, 'w') as f:
        yaml.dump(yaml_data, f, default_flow_style=False)
        
    print(f"Map generation complete! Files saved in: {DATASET_ROOT}")

if __name__ == "__main__":
    main()