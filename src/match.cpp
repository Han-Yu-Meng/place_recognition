// match.cpp

#include "lidar_simulator.hpp"
#include "sc_module.hpp"
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <pcl/common/transforms.h>
#include <pcl/console/print.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <thread>
#include <vector>
#include <vtkObject.h>

namespace fs = std::filesystem;

// [新增] 角度归一化辅助函数
inline double normalizeAngle(double rad) {
  while (rad > M_PI)
    rad -= 2.0 * M_PI;
  while (rad < -M_PI)
    rad += 2.0 * M_PI;
  return rad;
}

LidarSimulator::Pose matrixToPose(const Eigen::Matrix4d &T) {
  LidarSimulator::Pose p;
  p.x = T(0, 3);
  p.y = T(1, 3);
  p.z = T(2, 3);
  p.pitch = -asin(std::clamp(T(2, 0), -1.0, 1.0));
  if (abs(cos(p.pitch)) > 1e-5) {
    p.roll = atan2(T(2, 1), T(2, 2));
    p.yaw = atan2(T(1, 0), T(0, 0));
  } else {
    p.roll = atan2(-T(1, 2), T(1, 1));
    p.yaw = 0.0;
  }
  return p;
}

// 辅助函数：离群点滤除
void filterOutliers(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud) {
  if (!cloud || cloud->empty())
    return;

  // 确保输入输出分离，避免某些 PCL 版本下开启优化后的 aliasing 问题
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_v(new pcl::PointCloud<pcl::PointXYZ>);
  
  // 使用堆分配 (Heap Allocation) 防止 O2 优化下的栈对齐问题 (Stack Alignment Issues)
  auto vox = std::make_shared<pcl::VoxelGrid<pcl::PointXYZ>>();
  vox->setInputCloud(cloud);
  vox->setLeafSize(0.15f, 0.15f, 0.15f);
  vox->filter(*cloud_v);

  if (cloud_v->empty()) {
    cloud = cloud_v; // 返回空云
    return;
  }

  auto sor = std::make_shared<pcl::StatisticalOutlierRemoval<pcl::PointXYZ>>();
  sor->setInputCloud(cloud_v);
  sor->setMeanK(20);
  sor->setStddevMulThresh(2.0);
  
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_sor(new pcl::PointCloud<pcl::PointXYZ>);
  sor->filter(*cloud_sor);
  
  cloud = cloud_sor;
}

struct KeyFrame {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  int id;
  std::string pcd_path;
  Eigen::Matrix4d pose;
  bool is_360_fov;
};

int main(int argc, char **argv) {
  pcl::console::setVerbosityLevel(pcl::console::L_ERROR);
  vtkObject::GlobalWarningDisplayOff();

  std::vector<std::string> database_dirs = {
      // "/home/steven/Data/place_recognition/grid_features/"
      "/home/steven/Data/place_recognition/features/"};
  std::vector<std::string> query_dirs = {
      "/home/steven/Data/place_recognition/features/"};

  std::string map_path = "/home/steven/Data/place_recognition/global_map.pcd";

  auto sc_manager = std::make_unique<SCManager>();
  auto simulator = std::make_unique<LidarSimulator>();

  if (!simulator->load_map(map_path, 0.05)) {
    return -1;
  }

  // 加载 KeyFrames
  auto load_from_dirs = [&](const std::vector<std::string> &dirs) {
    std::vector<KeyFrame> kfs;
    std::vector<std::string> files;
    for (const auto &dir : dirs) {
      if (!fs::exists(dir))
        continue;
      for (const auto &entry : fs::directory_iterator(dir)) {
        if (entry.path().extension() == ".pcd") {
          files.push_back(entry.path().string());
        }
      }
    }
    std::sort(files.begin(), files.end());

    for (const auto &pcd_path : files) {
      std::string odom_path =
          pcd_path.substr(0, pcd_path.find_last_of('.')) + ".odom";
      Eigen::Matrix4d pose = Eigen::Matrix4d::Identity();
      if (fs::exists(odom_path)) {
        std::ifstream file(odom_path);
        for (int r = 0; r < 4; ++r)
          for (int c = 0; c < 4; ++c)
            file >> pose(r, c);
      }
      kfs.push_back({std::stoi(fs::path(pcd_path).stem().string()), pcd_path,
                     pose, false});
    }
    return kfs;
  };

  std::cout << "[Main] Loading Database Frames..." << std::endl;
  std::vector<KeyFrame> database_keyframes_all = load_from_dirs(database_dirs);
  std::vector<KeyFrame> database_keyframes;

  // 构建数据库
  for (const auto &kf : database_keyframes_all) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(
        new pcl::PointCloud<pcl::PointXYZ>);
    if (pcl::io::loadPCDFile(kf.pcd_path, *cloud) != -1) {
      // filterOutliers(cloud);
      sc_manager->setUseFovMask(!kf.is_360_fov);
      if (sc_manager->makeAndSaveScancontextAndKeys(*cloud)) {
        database_keyframes.push_back(kf);
      }
    }
  }

  std::cout << "[Main] Loading Query Poses..." << std::endl;
  std::vector<KeyFrame> query_keyframes = load_from_dirs(query_dirs);

  std::cout << "[Main] Database size: " << database_keyframes.size()
            << std::endl;
  std::cout << "[Main] Query set size: " << query_keyframes.size() << std::endl;

  if (database_keyframes.empty() || query_keyframes.empty())
    return -1;

  int correct_matches = 0, incorrect_matches = 0, not_found = 0,
      total_tests = 0;
  std::string output_dir = "/home/steven/Data/place_recognition/matchs/";
  std::string failure_dir = "/home/steven/Data/place_recognition/failures/";
  fs::create_directories(output_dir);
  fs::create_directories(failure_dir);

  std::cout << "\n[Main] Starting Simulation & Verification..." << std::endl;
  std::cout << "| " << std::setw(6) << "GT ID"
            << " | " << std::setw(8) << "Match ID"
            << " | " << std::setw(12) << "Est Yaw"
            << " | " << std::setw(12) << "Ang Err"
            << " | " << std::setw(8) << "Dist(m)"
            << " | " << std::setw(8) << "Sim(ms)"
            << " | " << std::setw(8) << "Det(ms)"
            << " | " << std::setw(6) << "Status"
            << " |" << std::endl;

  for (const auto &kf : query_keyframes) {
    auto sim_pose = matrixToPose(kf.pose);
    
    auto t1 = std::chrono::steady_clock::now();
    auto sim_cloud = simulator->simulate_scan(sim_pose);
    auto t2 = std::chrono::steady_clock::now();
    double sim_time = std::chrono::duration<double, std::milli>(t2 - t1).count();

    if (sim_cloud->empty())
      continue;
    // filterOutliers(sim_cloud);

    sc_manager->setUseFovMask(true); // Query 默认有盲区
    
    auto t3 = std::chrono::steady_clock::now();
    auto result = sc_manager->detectLoopClosureID(sim_cloud);
    auto t4 = std::chrono::steady_clock::now();
    double det_time = std::chrono::duration<double, std::milli>(t4 - t3).count();

    // --- 状态判定 ---
    std::string s_out = "FAIL";
    double dist = -1.0, angle_error = -1.0;

    if (result.first == -1) {
      s_out = "N/A";
      not_found++;
    } else {
      Eigen::Vector3d gt_pos = kf.pose.block<3, 1>(0, 3);
      Eigen::Vector3d match_pos =
          database_keyframes[result.first].pose.block<3, 1>(0, 3);
      dist = (gt_pos - match_pos).norm();

      double query_yaw = sim_pose.yaw;
      double match_yaw =
          matrixToPose(database_keyframes[result.first].pose).yaw;
      double gt_yaw_diff = normalizeAngle(match_yaw - query_yaw);
      double est_yaw_diff = result.second;
      angle_error =
          std::abs(normalizeAngle(est_yaw_diff - gt_yaw_diff)) * 180.0 / M_PI;

      if (dist <= 7.0 && angle_error < 30.0) {
        s_out = "OK";
        correct_matches++;
      } else {
        incorrect_matches++;
      }
    }
    total_tests++;

    // --- 打印日志 ---
    std::cout << "| " << std::setw(6) << kf.id << " | " << std::setw(8)
              << result.first << " | " << std::setw(12) << std::fixed
              << std::setprecision(2) << (result.second * 180.0 / M_PI)
              << " | ";
    if (result.first == -1)
      std::cout << std::setw(12) << "---"
                << " | " << std::setw(8) << "---";
    else
      std::cout << std::setw(12) << angle_error << " | " << std::setw(8)
                << dist;
    
    std::cout << " | " << std::setw(8) << sim_time << " | " << std::setw(8) << det_time;
    std::cout << " | " << std::setw(6) << s_out << " |" << std::endl;

    // --- FAILED 状态下的 BEV 可视化 ---
    if (s_out == "FAIL") {
      // 1. 加载真值点云 (Ground Truth)
      pcl::PointCloud<pcl::PointXYZ>::Ptr true_cloud(
          new pcl::PointCloud<pcl::PointXYZ>);
      if (fs::exists(kf.pcd_path)) {
        if (pcl::io::loadPCDFile(kf.pcd_path, *true_cloud) != -1) {
          filterOutliers(true_cloud);
        }
      }

      // 2. 加载匹配到的点云 (Match)
      pcl::PointCloud<pcl::PointXYZ>::Ptr match_cloud(
          new pcl::PointCloud<pcl::PointXYZ>);
      if (result.first != -1) {
        if (pcl::io::loadPCDFile(database_keyframes[result.first].pcd_path,
                                 *match_cloud) != -1) {
          filterOutliers(match_cloud);
        }
      }

      // 3. 显示调试界面
      sc_manager->showTripletDebug(sim_cloud, true_cloud, match_cloud,
                                   "SC Debug [FAIL]");

      // [新增] 保存调试信息到 failures 目录
      std::string base_fail = failure_dir + std::to_string(kf.id);
      
      // A. 保存可视化图片 (使用 PNG 保证清晰度)
      cv::imwrite(base_fail + "_bev.png", sc_manager->getCombinedBEVDebugView(sim_cloud, true_cloud, match_cloud));
      cv::imwrite(base_fail + "_sc.png", sc_manager->getCombinedSCDebugView(sim_cloud, true_cloud, match_cloud));

      // B. 保存三合一点云 (Query: Green, GT: Red, Match: Blue)
      pcl::PointCloud<pcl::PointXYZRGB>::Ptr tri_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
      auto add_to_tri = [&](pcl::PointCloud<pcl::PointXYZ>::Ptr src, uint8_t r, uint8_t g, uint8_t b) {
        if (!src) return;
        for (const auto &p : src->points) {
          pcl::PointXYZRGB pr;
          pr.x = p.x; pr.y = p.y; pr.z = p.z;
          pr.r = r; pr.g = g; pr.b = b;
          tri_cloud->push_back(pr);
        }
      };
      add_to_tri(sim_cloud, 0, 255, 0);    // Green
      add_to_tri(true_cloud, 255, 0, 0);   // Red
      add_to_tri(match_cloud, 0, 0, 255);  // Blue
      if (!tri_cloud->empty()) {
        pcl::io::savePCDFileBinary(base_fail + "_triplet.pcd", *tri_cloud);
      }

      std::cout << "    >>> [Failure Saved] ID: " << kf.id << " to " << failure_dir << std::endl;
      cv::waitKey(1);
    }
  }

  // ... 统计输出 ...

  std::cout << "\n[Main] Matching Summary:" << std::endl;
  std::cout << "    Total Queries: " << total_tests << std::endl;
  std::cout << "    Correct Matches: " << correct_matches << std::endl;
  std::cout << "    Incorrect Matches: " << incorrect_matches << std::endl;
  std::cout << "    Not Found: " << not_found << std::endl;
  std::cout << "    Accuracy: " << std::fixed << std::setprecision(2)
            << (100.0 * correct_matches / total_tests) << "%" << std::endl;
  return 0;
}