// grid_feature_extractor.cpp

#include "lidar_simulator.hpp"
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <fstream>

namespace fs = std::filesystem;

// 全局变量用于鼠标回调
LidarSimulator *g_simulator;
std::string g_output_dir;
int g_id_counter = 100000;
double g_min_x, g_max_x, g_min_y, g_max_y;
int g_img_w = 1000, g_img_h = 1000;
float g_scale;
double g_robot_height = 0.325;
cv::Mat g_display_map;

// 鼠标点击回调函数
void onMouse(int event, int x, int y, int flags, void *userdata) {
  if (event != cv::EVENT_LBUTTONDOWN)
    return;

  // 1. 像素坐标转世界坐标
  double world_x = g_min_x + (x - 50) / g_scale;
  double world_y = g_min_y + (g_img_h - 50 - y) / g_scale;

  std::cout << "\n[Interactive] Selected Position: (" << std::fixed
            << std::setprecision(2) << world_x << ", " << world_y << ")"
            << std::endl;

  // 2. 检查地面高度
  double ground_z;
  if (!g_simulator->find_ground_height(world_x, world_y, ground_z, 2.0)) {
    std::cout << "[Warning] Could not find valid ground height here!"
              << std::endl;
    return;
  }

  if (ground_z + g_robot_height > 2.0) {
    std::cout << "[Warning] Ground height is too high (z=" << ground_z
              << "), likely on a shelf or ceiling. Simulation aborted."
              << std::endl;
    return;
  }

  std::cout << "[Interactive] Ground found at z=" << ground_z
            << ". Simulating 8 directions..." << std::endl;

  // 3. 模拟 8 个方向
  bool any_saved = false;
  for (int yaw_deg = 0; yaw_deg < 360; yaw_deg += 45) {
    LidarSimulator::Pose pose;
    pose.x = world_x;
    pose.y = world_y;
    pose.z = ground_z + g_robot_height;
    pose.roll = 0;
    pose.pitch = 0;
    pose.yaw = yaw_deg * M_PI / 180.0;

    auto full_scan = g_simulator->simulate_scan(pose);

    if (full_scan->points.size() < 1000) {
      std::cout << "  -> Yaw " << yaw_deg << " deg: Scan too sparse ("
                << full_scan->points.size() << " pts), skipping." << std::endl;
      continue;
    }

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
        Eigen::AngleAxisd(pose.yaw, Eigen::Vector3d::UnitZ()).toRotationMatrix();
    odom_file << R(0, 0) << " " << R(0, 1) << " " << R(0, 2) << " " << pose.x
              << "\n";
    odom_file << R(1, 0) << " " << R(1, 1) << " " << R(1, 2) << " " << pose.y
              << "\n";
    odom_file << R(2, 0) << " " << R(2, 1) << " " << R(2, 2) << " " << pose.z
              << "\n";
    odom_file << 0.0 << " " << 0.0 << " " << 0.0 << " " << 1.0 << "\n";
    odom_file.close();

    std::cout << "  -> Saved: " << base_name << " (Yaw: " << yaw_deg << ")"
              << std::endl;
    g_id_counter++;
    any_saved = true;
  }

  // 4. 在界面上标记该点
  cv::Scalar color = any_saved ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255);
  cv::circle(g_display_map, cv::Point(x, y), 4, color, -1);
  cv::imshow("Grid Feature Extractor - Interactive", g_display_map);
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

  double min_z_m, max_z_m;
  simulator.get_map_bounds(g_min_x, g_max_x, g_min_y, g_max_y, min_z_m, max_z_m);

  // 计算缩放比例，渲染 BEV 地图
  float scale_x = (g_img_w - 100) / (g_max_x - g_min_x);
  float scale_y = (g_img_h - 100) / (g_max_y - g_min_y);
  g_scale = std::min(scale_x, scale_y);

  g_display_map = cv::Mat::zeros(g_img_h, g_img_w, CV_8UC3);
  std::cout << "[Grid] Generating BEV Map Preview..." << std::endl;

  // 渲染地图点云到 BEV 界面 (降采样以提高速度)
  for (size_t i = 0; i < simulator.global_map_->size(); i += 20) {
    auto &pt = simulator.global_map_->points[i];
    int ix = 50 + (int)((pt.x - g_min_x) * g_scale);
    int iy = g_img_h - 50 - (int)((pt.y - g_min_y) * g_scale);
    if (ix >= 0 && ix < g_img_w && iy >= 0 && iy < g_img_h) {
      // 根据高度着色，增强视觉效果
      uchar brightness = (uchar)std::clamp(
          (pt.z - min_z_m) / (max_z_m - min_z_m) * 200 + 55, 55.0, 255.0);
      g_display_map.at<cv::Vec3b>(iy, ix) =
          cv::Vec3b(brightness / 2, brightness, brightness / 2);
    }
  }

  std::cout << "\n==============================================" << std::endl;
  std::cout << "  INTERACTIVE GRID FEATURE EXTRACTOR" << std::endl;
  std::cout << "----------------------------------------------" << std::endl;
  std::cout << "  1. Click on the map to select a point." << std::endl;
  std::cout << "  2. The system will simulate 8 views." << std::endl;
  std::cout << "  3. Green dot = Success, Red dot = Failed." << std::endl;
  std::cout << "  4. Press ESC or 'q' to quit." << std::endl;
  std::cout << "==============================================\n" << std::endl;

  cv::namedWindow("Grid Feature Extractor - Interactive", cv::WINDOW_NORMAL);
  cv::setMouseCallback("Grid Feature Extractor - Interactive", onMouse, NULL);

  while (true) {
    cv::imshow("Grid Feature Extractor - Interactive", g_display_map);
    char key = (char)cv::waitKey(10);
    if (key == 27 || key == 'q')
      break;
  }

  std::cout << "[Grid] Session closed." << std::endl;
  return 0;
}
