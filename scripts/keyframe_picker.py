import sys
import os
import json
import numpy as np
import open3d as o3d
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QLabel
from PyQt5.QtGui import QPainter, QColor, QPen, QTransform, QWheelEvent, QMouseEvent
from PyQt5.QtCore import Qt, QPointF, QRectF

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'DATASET'), 'r') as f:
    DATASET_NAME = f.read().strip()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PCD_PATH = os.path.join(BASE_DIR, DATASET_NAME, "global_map.pcd")
OUTPUT_JSON = os.path.join(BASE_DIR, DATASET_NAME, "tasks.json")

class MapWidget(QWidget):
    def __init__(self, points, tasks, output_json, parent=None):
        super().__init__(parent)
        self.points = points  # nx3 numpy array (x, y, z)
        self.tasks = tasks    # list of dicts {"world_x": x, "world_y": y}
        self.output_json = output_json
        
        # 预计算点云的范围 (平面视图)
        self.min_x, self.min_y = np.min(self.points[:, :2], axis=0)
        self.max_x, self.max_y = np.max(self.points[:, :2], axis=0)
        self.center_x = (self.min_x + self.max_x) / 2
        self.center_y = (self.min_y + self.max_y) / 2
        
        # 渲染参数
        self.scale = 20.0  # pixels per meter
        self.offset_x = 0
        self.offset_y = 0
        
        self.last_mouse_pos = None
        self.setMouseTracking(True)

    def world_to_screen(self, x, y):
        # 俯视图：x 对应 Qt 的 x，y 对应 Qt 的 -y (Qt 坐标系 y 向下)
        screen_x = (x - self.center_x) * self.scale + self.width() / 2 + self.offset_x
        screen_y = -(y - self.center_y) * self.scale + self.height() / 2 + self.offset_y
        return screen_x, screen_y

    def screen_to_world(self, sx, sy):
        world_x = (sx - self.width() / 2 - self.offset_x) / self.scale + self.center_x
        world_y = -(sy - self.height() / 2 - self.offset_y) / self.scale + self.center_y
        return world_x, world_y

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(30, 30, 30))
        
        # 只有在点云存在时渲染
        if len(self.points) == 0:
            return

        # 获取高度 z 的范围用于映射颜色
        z_min, z_max = np.min(self.points[:, 2]), np.max(self.points[:, 2])
        z_range = max(z_max - z_min, 0.1)

        # 批量计算屏幕坐标以提高效率
        s_coords_x = (self.points[:, 0] - self.center_x) * self.scale + self.width() / 2 + self.offset_x
        s_coords_y = -(self.points[:, 1] - self.center_y) * self.scale + self.height() / 2 + self.offset_y
        
        # 筛选窗口内的点
        w, h = self.width(), self.height()
        mask = (s_coords_x >= 0) & (s_coords_x < w) & (s_coords_y >= 0) & (s_coords_y < h)
        
        vis_pts_x = s_coords_x[mask]
        vis_pts_y = s_coords_y[mask]
        vis_pts_z = self.points[mask, 2]

        # 抽样显示以提高帧率 (如果点云很大)
        stride = max(1, len(vis_pts_x) // 50000)
        
        for i in range(0, len(vis_pts_x), stride):
            # 高度 -> 颜色映射 (蓝 -> 绿 -> 红)
            norm_z = (vis_pts_z[i] - z_min) / z_range
            color = QColor.fromHsvF(0.66 * (1.0 - norm_z), 0.8, 0.9)
            painter.setPen(color)
            painter.drawPoint(int(vis_pts_x[i]), int(vis_pts_y[i]))

        # 绘制已选任务点
        painter.setPen(QPen(Qt.red, 3))
        for task in self.tasks:
            sx, sy = self.world_to_screen(task["world_x"], task["world_y"])
            painter.setBrush(Qt.red)
            painter.drawEllipse(QPointF(sx, sy), 5, 5)

    def wheelEvent(self, event: QWheelEvent):
        # 缩放
        delta = event.angleDelta().y()
        zoom_factor = 1.1 if delta > 0 else 0.9
        
        # 保持鼠标所在位置的世界坐标不变
        mouse_world_x, mouse_world_y = self.screen_to_world(event.x(), event.y())
        
        self.scale *= zoom_factor
        
        # 重新计算偏移以实现锚点缩放
        new_sx, new_sy = self.world_to_screen(mouse_world_x, mouse_world_y)
        self.offset_x -= (new_sx - event.x())
        self.offset_y -= (new_sy - event.y())
        
        self.update()

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            # Left Click: Pick point
            wx, wy = self.screen_to_world(event.x(), event.y())
            self._add_task(wx, wy)
        elif event.button() == Qt.MidButton:
            # Middle Click: Start Panning
            self.last_mouse_pos = event.pos()
        elif event.button() == Qt.RightButton:
            # Right Click: Delete nearest or last point
            self._delete_nearest(event.x(), event.y())

    def mouseMoveEvent(self, event: QMouseEvent):
        if event.buttons() & Qt.MidButton:
            if self.last_mouse_pos:
                diff = event.pos() - self.last_mouse_pos
                self.offset_x += diff.x()
                self.offset_y += diff.y()
                self.last_mouse_pos = event.pos()
                self.update()

    def _delete_nearest(self, sx, sy):
        if not self.tasks:
            return
        
        # 寻找距离点击位置最近的任务点
        idx_to_remove = -1
        min_dist = 20  # 像素阈值，点击 20 像素内才删除
        
        for i, task in enumerate(self.tasks):
            tx, ty = self.world_to_screen(task["world_x"], task["world_y"])
            dist = np.sqrt((tx - sx)**2 + (ty - sy)**2)
            if dist < min_dist:
                min_dist = dist
                idx_to_remove = i
        
        if idx_to_remove != -1:
            self.tasks.pop(idx_to_remove)
            self._save_json()
            self.update()
            print(f"[Frontend] Deleted task at index {idx_to_remove}")
        else:
            # 如果没点到某个点，默认删除最后一个
            self.tasks.pop()
            self._save_json()
            self.update()
            print("[Frontend] No point near click, undo last pick")

    def _add_task(self, wx, wy):
        task = {"world_x": float(wx), "world_y": float(wy)}
        self.tasks.append(task)
        self._save_json()
        self.update()
        print(f"[Frontend] Picked: {task}")

    def _undo(self):
        if self.tasks:
            self.tasks.pop()
            self._save_json()
            self.update()
            print("[Frontend] Undo last pick")

    def _save_json(self):
        os.makedirs(os.path.dirname(self.output_json), exist_ok=True)
        with open(self.output_json, 'w') as f:
            json.dump(self.tasks, f, indent=4)

class MainWindow(QMainWindow):
    def __init__(self, pcd_path, output_json):
        super().__init__()
        self.setWindowTitle("Keyframe Picker (Qt 2D View)")
        self.resize(1000, 800)
        
        # 加载点云
        print(f"[Frontend] Loading Point Cloud from {pcd_path}...")
        pcd = o3d.io.read_point_cloud(pcd_path)
        points = np.asarray(pcd.points)
        
        # 去掉 3m 以上的点云 (过滤 z > 3.0)
        mask = points[:, 2] <= 3.0
        filtered_points = points[mask]
        print(f"[Frontend] Removed {len(points) - len(filtered_points)} points above 3m.")
        
        # 加载任务
        tasks = []
        if os.path.exists(output_json):
            try:
                with open(output_json, 'r') as f:
                    tasks = json.load(f)
                print(f"[Frontend] Loaded {len(tasks)} existing tasks.")
            except Exception as e:
                print(f"[Warning] Failed to load tasks: {e}")
                
        self.map_widget = MapWidget(filtered_points, tasks, output_json)
        
        central_widget = QWidget()
        layout = QVBoxLayout(central_widget)
        info = QLabel("左键: 选点 | 滚轮: 缩放 | 滚轮按下: 拖动 | 右键: 删除 (最近点/末尾)")
        info.setAlignment(Qt.AlignCenter)
        info.setFixedHeight(30)
        info.setStyleSheet("color: white; background-color: #333333; font-weight: bold;")
        layout.addWidget(info)
        layout.addWidget(self.map_widget)
        layout.setContentsMargins(0,0,0,0)
        layout.setSpacing(0)
        
        self.setCentralWidget(central_widget)
        self.setStyleSheet("QMainWindow { background-color: #1e1e1e; }")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    if not os.path.exists(PCD_PATH):
        print(f"[Error] File not found: {PCD_PATH}")
        sys.exit(1)
    window = MainWindow(PCD_PATH, OUTPUT_JSON)
    window.show()
    sys.exit(app.exec_())
