// grid_feature_extractor.cpp

#include "lidar_simulator.hpp"
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

namespace fs = std::filesystem;

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <queue>
#include <thread>

namespace fs = std::filesystem;

// 异步任务处理类
struct ExtractTask {
  double world_x, world_y;
  int display_x, display_y;
};

// 全局变量用于鼠标回调
LidarSimulator *g_simulator;
std::string g_output_dir;
int g_id_counter = 100000;
double g_min_x, g_max_x, g_min_y, g_max_y;
int g_img_w = 1000, g_img_h = 1000;
float g_scale;
float g_base_scale;         // 每一级别的基准缩放
float g_zoom_factor = 1.0f; // 当前缩放因子
double g_offset_x = 0.0;    // 视图偏移 X
double g_offset_y = 0.0;    // 视图偏移 Y
double g_robot_height = 0.325;
cv::Mat g_base_map;    // 原始全图 (用于缩放)
cv::Mat g_display_map; // 当前显示的图
std::mutex g_display_mutex;
std::vector<cv::Point2d> g_trajectory;

std::queue<ExtractTask> g_task_queue;
std::mutex g_queue_mutex;
std::condition_variable g_queue_cv;
std::atomic<bool> g_running{true};

// 坐标转换工具
void worldToScreen(double wx, double wy, int &sx, int &sy) {
  sx = (int)((wx - g_min_x) * g_scale * g_zoom_factor + g_offset_x);
  sy = g_img_h - (int)((wy - g_min_y) * g_scale * g_zoom_factor + g_offset_y);
}

void screenToWorld(int sx, int sy, double &wx, double &wy) {
  wx = g_min_x + (sx - g_offset_x) / (g_scale * g_zoom_factor);
  wy = g_min_y + (g_img_h - sy - g_offset_y) / (g_scale * g_zoom_factor);
}

// 刷新显示
void updateDisplay() {
  std::lock_guard<std::mutex> lock(g_display_mutex);

  // 从 base_map 截取并缩放
  // 简单实现：重绘整个地图太慢，我们直接基于点云重绘或者操作 Image
  // 为了平滑缩放，这里每次都重新基于点云投影可能太慢，建议基于 g_base_map
  // 做仿射变换

  cv::Mat view = cv::Mat::zeros(g_img_h, g_img_w, CV_8UC3);

  // 构建变换矩阵
  // World -> Screen:
  // x_s = (x_w - min_x) * s * z + off_x
  // y_s = H - ((y_w - min_y) * s * z + off_y)

  // 这里为了效率，我们利用 OpenCV 的 warpAffine 对 g_base_map 进行变换
  // 但 g_base_map 分辨率可能不够，缩放变大时会模糊。为了高质量，还是重绘点吧。
  // 但是点云量大时会卡。对于 1000x1000 窗口，重绘 10 万点非常快。

  if (g_simulator && g_simulator->global_map_) {
    // 多线程渲染或降采样? 为了响应速度，这里只渲染视口内的点
    double visible_min_x, visible_max_x, visible_min_y, visible_max_y;
    screenToWorld(0, g_img_h, visible_min_x, visible_min_y);
    screenToWorld(g_img_w, 0, visible_max_x, visible_max_y);

    // 简单扩一点边界
    visible_min_x -= 5.0;
    visible_max_x += 5.0;
    visible_min_y -= 5.0;
    visible_max_y += 5.0;

    // 直接遍历所有点 (几十万点在 C++ 遍历很快，主要是 draw 耗时)
    // 优化：预先计算 min_z, max_z
    static double min_z_m = -100, max_z_m = 100;
    static bool range_set = false;
    if (!range_set) {
      double dummy1; // Use double to match function signature
      g_simulator->get_map_bounds(dummy1, dummy1, dummy1, dummy1, min_z_m,
                                  max_z_m);
      range_set = true;
    }

    // 绘制效率优化：直接操作像素指针
    for (const auto &pt : g_simulator->global_map_->points) {
      if (pt.x < visible_min_x || pt.x > visible_max_x ||
          pt.y < visible_min_y || pt.y > visible_max_y)
        continue;

      int sx, sy;
      worldToScreen(pt.x, pt.y, sx, sy);

      if (sx >= 0 && sx < g_img_w && sy >= 0 && sy < g_img_h) {
        uchar brightness = (uchar)std::clamp(
            (pt.z - min_z_m) / (max_z_m - min_z_m) * 200 + 55, 55.0, 255.0);
        view.at<cv::Vec3b>(sy, sx) =
            cv::Vec3b(brightness / 2, brightness, brightness / 2);
      }
    }
  }

  // Draw Trajectory
  if (g_trajectory.size() >= 2) {
    for (size_t i = 1; i < g_trajectory.size(); ++i) {
        int sx1, sy1, sx2, sy2;
        worldToScreen(g_trajectory[i-1].x, g_trajectory[i-1].y, sx1, sy1);
        worldToScreen(g_trajectory[i].x, g_trajectory[i].y, sx2, sy2);
        
        cv::line(view, cv::Point(sx1, sy1), cv::Point(sx2, sy2), cv::Scalar(0, 0, 255), 2);
    }
  }

  g_display_map = view;
}

// 消费者线程：处理计算任务
void processTasks() {
  while (g_running) {
    ExtractTask task;
    {
      std::unique_lock<std::mutex> lock(g_queue_mutex);
      g_queue_cv.wait(lock, [] { return !g_task_queue.empty() || !g_running; });

      if (!g_running && g_task_queue.empty())
        return;

      task = g_task_queue.front();
      g_task_queue.pop();
    }

    // --- 执行耗时的 Ray Casting ---
    if (!g_simulator)
      continue;

    double ground_z;
    bool ground_found = g_simulator->find_ground_height(
        task.world_x, task.world_y, ground_z, 2.0);

    bool success = false;
    if (ground_found &&
        (ground_z + g_robot_height <= 2.5)) { // 稍微放宽一点高度check
      bool any_saved = false;
      for (int yaw_deg = 0; yaw_deg < 360; yaw_deg += 45) {
        LidarSimulator::Pose pose;
        pose.x = task.world_x;
        pose.y = task.world_y;
        pose.z = ground_z + g_robot_height;
        pose.roll = 0;
        pose.pitch = 0;
        pose.yaw = yaw_deg * M_PI / 180.0;

        auto full_scan = g_simulator->simulate_scan(pose);

        if (full_scan->points.size() < 500)
          continue; // 松一点

        std::stringstream ss;
        ss << std::setw(6) << std::setfill('0') << g_id_counter;
        std::string base_name = ss.str();
        std::string pcd_filename = g_output_dir + base_name + ".pcd";
        std::string odom_filename = g_output_dir + base_name + ".odom";

        pcl::io::savePCDFileBinary(pcd_filename, *full_scan);

        // 保存 Odom
        std::ofstream odom_file(odom_filename);
        odom_file << std::fixed << std::setprecision(6);
        Eigen::Matrix3d R =
            Eigen::AngleAxisd(pose.yaw, Eigen::Vector3d::UnitZ())
                .toRotationMatrix();
        odom_file << R(0, 0) << " " << R(0, 1) << " " << R(0, 2) << " "
                  << pose.x << "\n";
        odom_file << R(1, 0) << " " << R(1, 1) << " " << R(1, 2) << " "
                  << pose.y << "\n";
        odom_file << R(2, 0) << " " << R(2, 1) << " " << R(2, 2) << " "
                  << pose.z << "\n";
        odom_file << 0.0 << " " << 0.0 << " " << 0.0 << " " << 1.0 << "\n";
        odom_file.close();

        g_id_counter++;
        any_saved = true;
      }
      success = any_saved;
    }

    // --- 更新 UI 结果 ---
    {
      std::lock_guard<std::mutex> lock(g_display_mutex);
      // 这里我们不能直接画在 g_display_map 上，因为它随时被重绘
      // 应该不仅画在当前帧，最好有一个结果列表，updateDisplay 时统一画
      // 简单处理：我们只打印，并在下一次 updateDisplay
      // 时不体现（因为这是异步的） 更好的做法是维护一个 std::vector<ResultMark>
      // g_marks; 但为了简单，我们暂不保存永久标记，只在控制台输出完成
      if (success)
        std::cout << "[Async] Task Finished: ID Range saved." << std::endl;
      else
        std::cout << "[Async] Task Failed: No ground or scan invalid."
                  << std::endl;
    }
  }
}

// 鼠标点击回调函数
void onMouse(int event, int x, int y, int flags, void *userdata) {
  // 滚轮缩放
  if (event == cv::EVENT_MOUSEWHEEL) {
    float old_zoom = g_zoom_factor;
    if (cv::getMouseWheelDelta(flags) > 0) {
      g_zoom_factor *= 1.1f;
    } else {
      g_zoom_factor /= 1.1f;
    }
    if (g_zoom_factor < 0.1f)
      g_zoom_factor = 0.1f;
    if (g_zoom_factor > 20.0f)
      g_zoom_factor = 20.0f;

    // 保持鼠标指向的世界坐标不变
    // world_x = min_x + (x - off_x) / (s * z)
    // off_x_new = x - (world_x - min_x) * s * z_new

    // 1. Calculate world pos under mouse BEFORE zoom
    double wx, wy;
    screenToWorld(x, y, wx, wy);

    // 2. Calculate new offset to keep wx, wy at screen x, y
    g_offset_x = x - (wx - g_min_x) * g_scale * g_zoom_factor;
    g_offset_y = (g_img_h - y) - (wy - g_min_y) * g_scale * g_zoom_factor;

    updateDisplay();
    cv::imshow("Grid Feature Extractor - Interactive", g_display_map);
    return;
  }

  // 拖拽平移 (中键或右键)
  static int prev_x = -1, prev_y = -1;
  if (event == cv::EVENT_MBUTTONDOWN || event == cv::EVENT_RBUTTONDOWN) {
    prev_x = x;
    prev_y = y;
  } else if (event == cv::EVENT_MOUSEMOVE &&
             (flags & (cv::EVENT_FLAG_MBUTTON | cv::EVENT_FLAG_RBUTTON))) {
    g_offset_x += (x - prev_x);
    g_offset_y -= (y - prev_y); // y-axis inverted
    prev_x = x;
    prev_y = y;
    updateDisplay();
    cv::imshow("Grid Feature Extractor - Interactive", g_display_map);
  }

  // 左键点击添加任务
  if (event == cv::EVENT_LBUTTONDOWN) {
    double wx, wy;
    screenToWorld(x, y, wx, wy);

    // 立即在当前画面画个黄圈表示 "Processing"
    cv::circle(g_display_map, cv::Point(x, y), 5, cv::Scalar(0, 255, 255), 2);
    cv::imshow("Grid Feature Extractor - Interactive", g_display_map);

    // 加入队列
    ExtractTask task;
    task.world_x = wx;
    task.world_y = wy;
    task.display_x = x;
    task.display_y = y;

    {
      std::lock_guard<std::mutex> lock(g_queue_mutex);
      g_task_queue.push(task);
    }
    g_queue_cv.notify_one();

    std::cout << "[Interactive] Task queued for (" << wx << ", " << wy << ")"
              << std::endl;
  }
}

// 加载轨迹
void loadTrajectory() {
  std::string feat_dir = "/home/steven/Data/place_recognition/features/";
  if (!fs::exists(feat_dir)) return;

  std::vector<std::string> odom_files;
  for (const auto &entry : fs::directory_iterator(feat_dir)) {
    if (entry.path().extension() == ".odom") {
      odom_files.push_back(entry.path().string());
    }
  }
  std::sort(odom_files.begin(), odom_files.end());

  for (const auto &path : odom_files) {
    std::ifstream file(path);
    Eigen::Matrix4d pose = Eigen::Matrix4d::Identity();
    for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c)
            file >> pose(r, c);
    
    g_trajectory.push_back(cv::Point2d(pose(0, 3), pose(1, 3)));
  }
  std::cout << "[Grid] Loaded trajectory with " << g_trajectory.size() << " points." << std::endl;
}

int main(int argc, char **argv) {
  std::string map_path = "/home/steven/Data/place_recognition/global_map.pcd";
  g_output_dir = "/home/steven/Data/place_recognition/grid_features/";

  if (!fs::exists(g_output_dir)) {
    fs::create_directories(g_output_dir);
  }

  LidarSimulator simulator;
  if (!simulator.load_map(map_path, 0.05)) {
    return -1;
  }
  g_simulator = &simulator;

  loadTrajectory();

  double min_z_m, max_z_m;
  simulator.get_map_bounds(g_min_x, g_max_x, g_min_y, g_max_y, min_z_m,
                           max_z_m);

  // 计算初始缩放比例
  float scale_x = (g_img_w - 100.0f) / (g_max_x - g_min_x);
  float scale_y = (g_img_h - 100.0f) / (g_max_y - g_min_y);
  g_scale = std::min(scale_x, scale_y);

  // 初始偏移，居中
  g_offset_x = (g_img_w - (g_max_x - g_min_x) * g_scale) / 2.0;
  g_offset_y = (g_img_h - (g_max_y - g_min_y) * g_scale) / 2.0;

  g_display_map = cv::Mat::zeros(g_img_h, g_img_w, CV_8UC3);

  // 启动后台工作线程
  std::thread worker(processTasks);

  std::cout << "\n==============================================" << std::endl;
  std::cout << "  INTERACTIVE GRID FEATURE EXTRACTOR v2.0" << std::endl;
  std::cout << "----------------------------------------------" << std::endl;
  std::cout << "  [L-Click]   Queue Simulation Task" << std::endl;
  std::cout << "  [Wheel]     Zoom In/Out" << std::endl;
  std::cout << "  [R/M-Drag]  Pan View" << std::endl;
  std::cout << "  [ESC/Q]     Quit" << std::endl;
  std::cout << "==============================================\n" << std::endl;

  cv::namedWindow("Grid Feature Extractor - Interactive", cv::WINDOW_NORMAL);
  cv::setMouseCallback("Grid Feature Extractor - Interactive", onMouse, NULL);

  // 初始渲染
  updateDisplay();

  while (true) {
    if(g_display_map.rows > 0 && g_display_map.cols > 0)
        cv::imshow("Grid Feature Extractor - Interactive", g_display_map);
    char key = (char)cv::waitKey(30); // 30ms 刷新
    if (key == 27 || key == 'q')
      break;
  }

  g_running = false;
  g_queue_cv.notify_all();
  if (worker.joinable())
    worker.join();

  std::cout << "[Grid] Session closed." << std::endl;
  return 0;
}
