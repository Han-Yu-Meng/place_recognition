// simulator_test.cpp

#include "lidar_simulator.hpp"
#include <Eigen/Dense>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>

namespace fs = std::filesystem;

const std::string DATASET_DIR = "YunJing";

struct KeyFrame {
  int id;
  std::string pcd_path;
  Eigen::Matrix4d pose;
};

LidarSimulator::Pose matrixToPose(const Eigen::Matrix4d &T) {
  LidarSimulator::Pose p;
  p.x = T(0, 3);
  p.y = T(1, 3);
  p.z = T(2, 3);
  p.pitch = -asin(std::clamp(T(2, 0), -1.0, 1.0));
  if (std::abs(std::cos(p.pitch)) > 1e-5) {
    p.roll = std::atan2(T(2, 1), T(2, 2));
    p.yaw = std::atan2(T(1, 0), T(0, 0));
  } else {
    p.roll = std::atan2(-T(1, 2), T(1, 1));
    p.yaw = 0.0;
  }
  return p;
}

int main(int argc, char **argv) {
  fs::path root(PROJECT_ROOT_DIR);
  std::cout << "Project Root: " << PROJECT_ROOT_DIR << std::endl;
  fs::path dataset_dir = root / DATASET_DIR;
  fs::path processed_dir = dataset_dir / "keyframes" / "processed";
  fs::path compare_dir = dataset_dir / "keyframes" / "compare";

  if (!fs::exists(processed_dir)) {
    std::cerr << "Directory not found: " << processed_dir << std::endl;
    return -1;
  }

  if (!fs::exists(compare_dir)) {
    fs::create_directories(compare_dir);
  }

  // 1. 初始化模拟器
  LidarSimulator simulator;
  std::string config_path = (dataset_dir / "lidar_config.yaml").string();
  std::cout << "Loading Lidar Simulator with config: " << config_path << std::endl;
  simulator.load_config(config_path);
  if (!simulator.load_map((dataset_dir / "global_map.pcd").string(), 0.5)) {
    return -1;
  }

  // 2. 加载 processed 关键帧
  std::vector<KeyFrame> kfs;
  std::vector<std::string> pcd_files;
  for (const auto &entry : fs::directory_iterator(processed_dir)) {
    if (entry.path().extension() == ".pcd") {
      pcd_files.push_back(entry.path().string());
    }
  }
  std::sort(pcd_files.begin(), pcd_files.end());

  for (const auto &pcd_path : pcd_files) {
    std::string odom_path =
        pcd_path.substr(0, pcd_path.find_last_of('.')) + ".odom";
    if (fs::exists(odom_path)) {
      Eigen::Matrix4d pose = Eigen::Matrix4d::Identity();
      std::ifstream file(odom_path);
      for (int r = 0; r < 4; ++r)
        for (int c = 0; c < 4; ++c)
          file >> pose(r, c);
      kfs.push_back(
          {std::stoi(fs::path(pcd_path).stem().string()), pcd_path, pose});
    }
  }

  if (kfs.empty()) {
    std::cout << "No keyframes found in " << processed_dir << std::endl;
    return 0;
  }

  std::cout << "Generating comparison PCDs in: " << compare_dir << std::endl;

  for (size_t i = 0; i < kfs.size(); ++i) {
    const auto &kf = kfs[i];

    // 加载真实点云
    pcl::PointCloud<pcl::PointXYZ>::Ptr real_cloud(
        new pcl::PointCloud<pcl::PointXYZ>);
    if (pcl::io::loadPCDFile(kf.pcd_path, *real_cloud) == -1) {
      continue;
    }

    // 模拟点云
    LidarSimulator::Pose sim_pose = matrixToPose(kf.pose);
    auto sim_cloud = simulator.simulate_scan(sim_pose);

    // 合并点云并上色
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr compared_cloud(
        new pcl::PointCloud<pcl::PointXYZRGB>);

    // 真实点云：绿色 (0, 255, 0)
    for (const auto &p : real_cloud->points) {
      pcl::PointXYZRGB pr;
      pr.x = p.x; pr.y = p.y; pr.z = p.z;
      pr.r = 0; pr.g = 255; pr.b = 0;
      compared_cloud->points.push_back(pr);
    }

    // 模拟点云：红色 (255, 0, 0)
    for (const auto &p : sim_cloud->points) {
      pcl::PointXYZRGB pr;
      pr.x = p.x; pr.y = p.y; pr.z = p.z;
      pr.r = 255; pr.g = 0; pr.b = 0;
      compared_cloud->points.push_back(pr);
    }

    compared_cloud->width = compared_cloud->points.size();
    compared_cloud->height = 1;
    compared_cloud->is_dense = true;

    std::string out_path = (compare_dir / (fs::path(kf.pcd_path).stem().string() + "_compare.pcd")).string();
    pcl::io::savePCDFileBinary(out_path, *compared_cloud);

    if (i % 10 == 0 || i == kfs.size() - 1) {
      std::cout << "\r[Process] " << i + 1 << "/" << kfs.size() << " saved." << std::flush;
    }
  }

  std::cout << "\nAll comparison files saved." << std::endl;

  return 0;
}

