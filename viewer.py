import open3d as o3d
import numpy as np
import os
import glob
import sys

# ================= 配置 =================
FEATURES_DIR = "./features" # 特征文件夹路径
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720
# =======================================

class FeatureViewer:
    def __init__(self, feature_dir):
        self.feature_dir = feature_dir
        
        # 1. 获取所有 pcd 文件并排序
        self.pcd_files = sorted(glob.glob(os.path.join(feature_dir, "*.pcd")))
        if not self.pcd_files:
            print(f"[Error] No .pcd files found in {feature_dir}")
            sys.exit(1)
            
        self.num_frames = len(self.pcd_files)
        self.current_idx = 0
        self.view_mode_world = False # False=Local Frame, True=World Frame

        # 2. 初始化 Open3D 可视化器
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window(window_name="Feature Viewer (See Terminal for Info)", width=WINDOW_WIDTH, height=WINDOW_HEIGHT)
        
        # 3. 创建占位几何体
        self.pcd_geometry = o3d.geometry.PointCloud()
        self.axis_geometry = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        
        # 4. 注册按键回调
        self.vis.register_key_callback(262, self.next_frame)      # Right Arrow
        self.vis.register_key_callback(263, self.prev_frame)      # Left Arrow
        self.vis.register_key_callback(32,  self.toggle_view_mode) # Space

        # 5. 加载第一帧
        self.load_frame(self.current_idx)
        self.vis.add_geometry(self.pcd_geometry)
        self.vis.add_geometry(self.axis_geometry)
        
        # --- 关键修复：强制刷新一次，确保 ViewControl 被初始化 ---
        self.vis.poll_events()
        self.vis.update_renderer()

        # 设置初始视角
        ctr = self.vis.get_view_control()
        if ctr is not None:
            ctr.set_lookat([0, 0, 0])
            ctr.set_front([-1, 0, 1])
            ctr.set_up([0, 0, 1])
            ctr.set_zoom(0.8)
        else:
            print("[Warning] ViewControl is None. Graphics context might not be ready.")

        print(f"Viewer initialized with {self.num_frames} frames.")
        print("Controls:")
        print("  [Right Arrow] : Next Frame")
        print("  [Left Arrow]  : Previous Frame")
        print("  [Space Bar]   : Toggle Local/World View")
        print("  [Q]           : Quit")
        self.print_status()

    def load_frame(self, idx):
        """读取指定索引的 PCD 和 ODOM"""
        pcd_path = self.pcd_files[idx]
        odom_path = pcd_path.replace(".pcd", ".odom")
        
        cloud = o3d.io.read_point_cloud(pcd_path)
        
        pose = np.eye(4)
        self.has_pose = False
        if os.path.exists(odom_path):
            try:
                pose = np.loadtxt(odom_path)
                self.has_pose = True
            except:
                print(f"[Warn] Failed to load odom: {odom_path}")
        
        self.current_pose = pose 

        if self.view_mode_world:
            cloud.transform(pose)
            # 重建坐标轴
            self.axis_geometry = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
            self.axis_geometry.transform(pose)
        else:
            self.axis_geometry = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])

        self.pcd_geometry.points = cloud.points
        if cloud.has_colors():
            self.pcd_geometry.colors = cloud.colors
        
    def print_status(self):
        pcd_name = os.path.basename(self.pcd_files[self.current_idx])
        pos_str = f"Pos: [{self.current_pose[0,3]:.2f}, {self.current_pose[1,3]:.2f}, {self.current_pose[2,3]:.2f}]" if self.has_pose else "Pos: N/A"
        mode_str = "WORLD View" if self.view_mode_world else "LOCAL View"
        print(f"==> Frame [{self.current_idx+1}/{self.num_frames}] | {mode_str} | {pos_str} | ID: {pcd_name}")

    def update_view(self):
        self.vis.clear_geometries()
        self.vis.add_geometry(self.pcd_geometry)
        self.vis.add_geometry(self.axis_geometry)
        self.vis.poll_events()
        self.vis.update_renderer()
        self.print_status()

    def next_frame(self, vis):
        if self.current_idx < self.num_frames - 1:
            self.current_idx += 1
            self.load_frame(self.current_idx)
            self.update_view()
        return False

    def prev_frame(self, vis):
        if self.current_idx > 0:
            self.current_idx -= 1
            self.load_frame(self.current_idx)
            self.update_view()
        return False

    def toggle_view_mode(self, vis):
        self.view_mode_world = not self.view_mode_world
        self.load_frame(self.current_idx)
        
        if not self.view_mode_world:
            ctr = self.vis.get_view_control()
            if ctr is not None:
                ctr.set_lookat([0, 0, 0])
            
        self.update_view()
        return False

    def run(self):
        self.vis.run()
        self.vis.destroy_window()

if __name__ == "__main__":
    if not os.path.exists(FEATURES_DIR):
        print(f"Error: Directory {FEATURES_DIR} does not exist.")
    else:
        viewer = FeatureViewer(FEATURES_DIR)
        viewer.run()