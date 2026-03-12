import os
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R, Slerp
import rclpy
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
import rosbag2_py
import sensor_msgs_py.point_cloud2 as pc2

DATASET_NAME = 'YunJing'
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_ROOT = os.path.join(BASE_DIR, DATASET_NAME)

BAG_PATH = os.path.join(DATASET_ROOT, 'bag')
OUTPUT_DIR = os.path.join(DATASET_ROOT, 'keyframes', 'raw')

TOPIC_CLOUD = '/cloud_registered_body'
TOPIC_ODOM = '/Odometry'

# --- 关键帧配置 ---
# 这里的参数可以根据需要调整，或者从命令行参数获取
KEYFRAME_DIST = 0.2            # 关键帧距离阈值 (米)
KEYFRAME_ANGLE_DEG = 10.0      # 关键帧角度阈值 (度)

# --- 静态变换 (body -> lidar_odom) ---
# 参考 preprocess_deprecated.py 中的 T_BASE_TO_LIDAR
STATIC_TRANS = np.array([0.5496, 0.2400, 0.1934])
STATIC_QUAT = np.array([0.0, 0.130526, 0.0, 0.991445]) # x, y, z, w

# --- 过滤框配置 (基于 CAD 文件，包含 5cm 裕量) ---
FILTER_BOX_MIN = np.array([-0.25, -0.35, 0.05])
FILTER_BOX_MAX = np.array([0.55, 0.35, 0.87])

def get_static_tf():
    rot = R.from_quat(STATIC_QUAT).as_matrix()
    T = np.eye(4)
    T[:3, :3] = rot
    T[:3, 3] = STATIC_TRANS
    return T

T_BASE_TO_LIDAR = get_static_tf()

class OdomBuffer:
    def __init__(self, max_size=2000):
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
    
    xyz = np.array([(p[0], p[1], p[2]) for p in points])
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

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created output directory: {OUTPUT_DIR}")

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
    
    odom_buffer = OdomBuffer()
    last_keyframe_pose = None
    keyframe_count = 0
    
    print(f"Starting to process bag from: {BAG_PATH}")
    print(f"Output will be saved to: {OUTPUT_DIR}")

    while reader.has_next():
        (topic, data, t_ns) = reader.read_next()
        current_time_s = (t_ns - bag_start_time_ns) / 1e9
        
        if topic == TOPIC_ODOM:
            msg_type = get_message(type_map[topic])
            odom_msg = deserialize_message(data, msg_type)
            odom_buffer.add(current_time_s, odom_to_matrix(odom_msg))
            
        elif topic == TOPIC_CLOUD:
            msg_type = get_message(type_map[topic])
            cloud_msg = deserialize_message(data, msg_type)
            
            # 获取插值位姿
            T_curr = odom_buffer.get_interpolated_pose(current_time_s)
            if T_curr is None:
                continue

            # 关键帧判断逻辑
            is_keyframe = False
            if last_keyframe_pose is None:
                is_keyframe = True
            else:
                dist, angle = calculate_delta(last_keyframe_pose, T_curr)
                if dist >= KEYFRAME_DIST or angle >= KEYFRAME_ANGLE_DEG:
                    is_keyframe = True
            
            if is_keyframe:
                pcd = msg_to_pcd(cloud_msg)
                if pcd is None:
                    continue
                
                # 1. 坐标系转换 (Lidar -> Body)
                pcd.transform(T_BASE_TO_LIDAR)
                
                # 2. 滤除处于静态 Box 中的点 (移除机器人自身遮挡)
                points = np.asarray(pcd.points)
                mask = ~((points[:, 0] >= FILTER_BOX_MIN[0]) & (points[:, 0] <= FILTER_BOX_MAX[0]) &
                         (points[:, 1] >= FILTER_BOX_MIN[1]) & (points[:, 1] <= FILTER_BOX_MAX[1]) &
                         (points[:, 2] >= FILTER_BOX_MIN[2]) & (points[:, 2] <= FILTER_BOX_MAX[2]))
                pcd.points = o3d.utility.Vector3dVector(points[mask])

                T_ext = T_BASE_TO_LIDAR
                T_ext_inv = np.linalg.inv(T_ext)
                T_final_pose = T_ext @ T_curr @ T_ext_inv

                pcd_filename = os.path.join(OUTPUT_DIR, f"{keyframe_count:06d}.pcd")
                odom_filename = os.path.join(OUTPUT_DIR, f"{keyframe_count:06d}.odom")
                
                # 写文件
                o3d.io.write_point_cloud(pcd_filename, pcd)
                np.savetxt(odom_filename, T_final_pose)
                
                print(f"\r[KeyFrame] Saved {keyframe_count:06d} at {current_time_s:.2f}s", end='', flush=True)
                
                last_keyframe_pose = T_curr
                keyframe_count += 1

    print(f"\nFinished. Total keyframes: {keyframe_count}")

if __name__ == "__main__":
    main()
