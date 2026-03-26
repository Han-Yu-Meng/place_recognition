import os
import yaml
import json
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R, Slerp
import rclpy
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
import rosbag2_py
import sensor_msgs_py.point_cloud2 as pc2

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'DATASET'), 'r') as f:
    DATASET_NAME = f.read().strip()
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_ROOT = os.path.join(BASE_DIR, DATASET_NAME)

BAG_PATH = os.path.join(DATASET_ROOT, 'bag')
HBA_DIR = os.path.join(DATASET_ROOT, 'hba')
HBA_PCD_DIR = os.path.join(HBA_DIR, 'pcd')
HBA_POSE_FILE = os.path.join(HBA_DIR, 'pose.json')
CONFIG_PATH = os.path.join(DATASET_ROOT, 'lidar_config.yaml')

TOPIC_CLOUD = '/cloud_registered_body'
TOPIC_ODOM = '/Odometry'

# --- 关键帧配置 ---
KEYFRAME_DIST = 2.0            # 关键帧距离阈值 (米)
KEYFRAME_ANGLE_DEG = 10.0      # 关键帧角度阈值 (度)
MAX_FRAMES = 3400              # 最大生成的帧数，超过后停止处理

def load_lidar_config(path):
    with open(path, 'r') as f:
        content = f.read()
        if content.startswith('%YAML'):
            content = '\n'.join(content.split('\n')[1:])
        config = yaml.safe_load(content)
    
    T_baselink_lidar_list = config.get('T_baselink_lidar', [])
    if not T_baselink_lidar_list:
        raise ValueError(f"T_baselink_lidar not found in {path}")
    
    cfg = T_baselink_lidar_list[0]
    trans = np.array([cfg['x'], cfg['y'], cfg['z']])
    rot = R.from_euler('xyz', [cfg['roll'], cfg['pitch'], cfg['yaw']], degrees=True)
    quat = rot.as_quat() # x, y, z, w

    fb_min = np.array(config.get('filter_box_min', [-0.25, -0.35, 0.05]))
    fb_max = np.array(config.get('filter_box_max', [0.55, 0.35, 0.87]))

    return trans, quat, fb_min, fb_max

# --- 静态变换 (body -> lidar_odom) ---
STATIC_TRANS, STATIC_QUAT, FILTER_BOX_MIN, FILTER_BOX_MAX = load_lidar_config(CONFIG_PATH)

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
    if not os.path.exists(HBA_PCD_DIR):
        os.makedirs(HBA_PCD_DIR)
        print(f"Created HBA PCD directory: {HBA_PCD_DIR}")

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
    hba_poses = []
    
    print(f"Starting to process bag for HBA: {BAG_PATH}")

    while reader.has_next():
        if keyframe_count >= MAX_FRAMES:
            print(f"\nReached MAX_FRAMES ({MAX_FRAMES}). Stopping bag processing.")
            break

        (topic, data, t_ns) = reader.read_next()
        current_time_s = (t_ns - bag_start_time_ns) / 1e9
        
        if topic == TOPIC_ODOM:
            msg_type = get_message(type_map[topic])
            odom_msg = deserialize_message(data, msg_type)
            odom_buffer.add(current_time_s, odom_to_matrix(odom_msg))
            
        elif topic == TOPIC_CLOUD:
            msg_type = get_message(type_map[topic])
            cloud_msg = deserialize_message(data, msg_type)
            
            T_curr = odom_buffer.get_interpolated_pose(current_time_s)
            if T_curr is None:
                continue

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
                
                # 2. 滤除机器人自身遮挡
                points = np.asarray(pcd.points)
                mask = ~((points[:, 0] >= FILTER_BOX_MIN[0]) & (points[:, 0] <= FILTER_BOX_MAX[0]) &
                         (points[:, 1] >= FILTER_BOX_MIN[1]) & (points[:, 1] <= FILTER_BOX_MAX[1]) &
                         (points[:, 2] >= FILTER_BOX_MIN[2]) & (points[:, 2] <= FILTER_BOX_MAX[2]))
                pcd.points = o3d.utility.Vector3dVector(points[mask])

                # 计算 Body 坐标系下的位姿
                T_ext = T_BASE_TO_LIDAR
                T_ext_inv = np.linalg.inv(T_ext)
                T_final_pose = T_ext @ T_curr @ T_ext_inv
                
                # 转换位姿格式: tx ty tz qw qx qy qz
                pos = T_final_pose[:3, 3]
                quat = R.from_matrix(T_final_pose[:3, :3]).as_quat() # [x, y, z, w]
                hba_pose_str = f"{pos[0]} {pos[1]} {pos[2]} {quat[3]} {quat[0]} {quat[1]} {quat[2]}"
                hba_poses.append(hba_pose_str)

                # 保存 PCD (使用 5 位数补全，例如 00000.pcd)
                pcd_filename = os.path.join(HBA_PCD_DIR, f"{keyframe_count:05d}.pcd")
                o3d.io.write_point_cloud(pcd_filename, pcd)
                
                print(f"\r[HBA] Saved {keyframe_count:05d} at {current_time_s:.2f}s", end='', flush=True)
                
                last_keyframe_pose = T_curr
                keyframe_count += 1

    # 保存 pose.json
    with open(HBA_POSE_FILE, 'w') as f:
        # HBA 期望的是每行一个位姿的格式。虽然叫 .json 但根据描述内容是文本列表
        # 这里为了符合 "pose.json file containing the initial poses" 的描述，
        # 我们按照用户提供的示例格式直接保存为文本行。
        f.write("\n".join(hba_poses))

    print(f"\nFinished. Total keyframes: {keyframe_count}")
    print(f"HBA data saved to: {HBA_DIR}")

if __name__ == "__main__":
    main()
