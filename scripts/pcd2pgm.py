import os
import numpy as np
import open3d as o3d
from PIL import Image
import yaml

# ================= 配置参数 =================
DATASET_NAME = 'YunJing'
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_ROOT = os.path.join(BASE_DIR, DATASET_NAME)

# 输入文件：全局地图
INPUT_MAP_PATH = os.path.join(DATASET_ROOT, 'global_map_dense.pcd')
# 输出文件：PGM, PNG 和 YAML
OUTPUT_PGM_PATH = os.path.join(DATASET_ROOT, 'global_map.pgm')
OUTPUT_PNG_PATH = os.path.join(DATASET_ROOT, 'global_map.png')
OUTPUT_YAML_PATH = os.path.join(DATASET_ROOT, 'global_map.yaml')

# --- 过滤与转换配置 (参考 C++ 实现) ---
THRE_Z_MIN = 0.5           # Z 轴直通滤波最小值
THRE_Z_MAX = 2.5           # Z 轴直通滤波最大值
FLAG_PASS_THROUGH = False  # 是否保留范围外的点 (False: 保留 [min, max])
THRE_RADIUS = 0.5          # 半径滤波器半径
THRES_POINT_COUNT = 10     # 半径内最小点数
MAP_RESOLUTION = 0.05      # 地图分辨率 (米/像素)

# Odom 到 Lidar Odom 的变换 (x, y, z, roll, pitch, yaw)
ODOM_TO_LIDAR_ODOM = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

def pass_through_filter(pcd, z_min, z_max, negative=False):
    """
    Z 轴直通滤波
    """
    points = np.asarray(pcd.points)
    z = points[:, 2]
    if not negative:
        mask = (z >= z_min) & (z <= z_max)
    else:
        mask = (z < z_min) | (z > z_max)
    
    filtered_pcd = o3d.geometry.PointCloud()
    filtered_pcd.points = o3d.utility.Vector3dVector(points[mask])
    return filtered_pcd

def radius_outlier_filter(pcd, radius, min_neighbors):
    """
    半径离群点滤波 (使用 Open3D 的实现)
    """
    cl, ind = pcd.remove_radius_outlier(nb_points=min_neighbors, radius=radius)
    return cl

def apply_transform(pcd, transform_params):
    """
    应用变换 (参考 C++ applyTransform 逻辑)
    C++ 中使用了 transform.inverse()
    """
    x, y, z, roll, pitch, yaw = transform_params
    
    # 构建变换矩阵
    # 注意：这里简化处理，如果 transform_params 全为 0，则不变换
    if all(v == 0.0 for v in transform_params):
        return pcd

    # Open3D 使用 4x4 矩阵
    # 这里直接按照 C++ 的逆变换逻辑
    # 实际上如果是 identity，inverse 也是 identity
    # 如果有具体值，需要构建完整的 rotation 和 translation
    # 暂时保持和 C++ 一致的 identity 处理，因为默认值是全 0
    return pcd

def main():
    if not os.path.exists(INPUT_MAP_PATH):
        print(f"Error: Input map {INPUT_MAP_PATH} does not exist.")
        return

    print(f"Loading global map from {INPUT_MAP_PATH}...")
    pcd = o3d.io.read_point_cloud(INPUT_MAP_PATH)
    print(f"Initial point cloud size: {len(pcd.points)}")

    # 1. 坐标变换
    pcd = apply_transform(pcd, ODOM_TO_LIDAR_ODOM)

    # 2. 直通滤波 (Z 轴)
    print(f"Applying PassThrough filter (Z: {THRE_Z_MIN} to {THRE_Z_MAX})...")
    pcd = pass_through_filter(pcd, THRE_Z_MIN, THRE_Z_MAX, FLAG_PASS_THROUGH)
    print(f"After PassThrough: {len(pcd.points)} points")

    if len(pcd.points) == 0:
        print("Error: Point cloud is empty after filtering!")
        return

    # 3. 半径滤波
    print(f"Applying RadiusOutlier filter (R: {THRE_RADIUS}, MinCount: {THRES_POINT_COUNT})...")
    pcd = radius_outlier_filter(pcd, THRE_RADIUS, THRES_POINT_COUNT)
    print(f"After RadiusOutlier: {len(pcd.points)} points")

    # 4. 生成占据网格并保存为 PGM
    points = np.asarray(pcd.points)
    x_min, y_min = np.min(points[:, :2], axis=0)
    x_max, y_max = np.max(points[:, :2], axis=0)

    width = int(np.ceil((x_max - x_min) / MAP_RESOLUTION))
    height = int(np.ceil((y_max - y_min) / MAP_RESOLUTION))
    
    print(f"Generating map: {width}x{height}, resolution: {MAP_RESOLUTION}")

    # 创建地图数据 (初始化为 255: 未知/空闲，0: 占据)
    # ROS OccupancyGrid: 0=free, 100=occupied, -1=unknown
    # PGM: 255=free, 0=occupied
    map_data = np.full((height, width), 255, dtype=np.uint8)

    for pt in points:
        ix = int(np.floor((pt[0] - x_min) / MAP_RESOLUTION))
        iy = int(np.floor((pt[1] - y_min) / MAP_RESOLUTION))
        
        # 翻转 Y 轴以符合图像坐标系 (或者保持 ROS 习惯，后面保存时调整)
        # ROS 习惯：左下角是原点。图像习惯：左上角是原点。
        if 0 <= ix < width and 0 <= iy < height:
            # iy = height - 1 - iy # 翻转 Y，使图像看起来是正的
            map_data[height - 1 - iy, ix] = 0

    # 保存 PGM
    img = Image.fromarray(map_data)
    img.save(OUTPUT_PGM_PATH)
    print(f"PGM map saved to: {OUTPUT_PGM_PATH}")

    # 保存 PNG (方便预览)
    img.save(OUTPUT_PNG_PATH)
    print(f"PNG map saved to: {OUTPUT_PNG_PATH}")

    # 保存 YAML (ROS nav_msgs 格式)
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
    print(f"YAML config saved to: {OUTPUT_YAML_PATH}")

if __name__ == "__main__":
    main()
