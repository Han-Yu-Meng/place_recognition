// match.cpp

#include "bev_debugger.hpp"
#include "lidar_simulator.hpp"
#include "sc_module.hpp"
#include <algorithm>
#include <filesystem>
#include <fstream>
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

// 辅助函数：离群点滤除 (优化版)
void filterOutliers(pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud) {
  if (cloud->empty())
    return;

  // 1. 先进行体素滤波，大幅减少点数，加速后续处理且不影响 ScanContext 结构
  pcl::VoxelGrid<pcl::PointXYZ> vox;
  vox.setInputCloud(cloud);
  vox.setLeafSize(0.15f, 0.15f, 0.15f);
  pcl::PointCloud<pcl::PointXYZ>::Ptr temp_cloud(
      new pcl::PointCloud<pcl::PointXYZ>);
  vox.filter(*temp_cloud);

  // 2. 较宽松的统计滤波
  pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
  sor.setInputCloud(temp_cloud);
  sor.setMeanK(20);            // 之前的 50 太慢了，减少计算邻域
  sor.setStddevMulThresh(2.0); // 之前的 1.0 太严了，增加阈值到 3.0
  sor.filter(*cloud);
}

struct KeyFrame {
  int id;
  std::string pcd_path;
  Eigen::Matrix4d pose;
  bool is_360_fov; // [新增] 标记是否为 360 度 FOV
};

int main(int argc, char **argv) {
  pcl::console::setVerbosityLevel(pcl::console::L_ERROR);
  vtkObject::GlobalWarningDisplayOff();

  // [修改] 分别指定数据库文件夹和查询（模拟位姿）文件夹
  std::vector<std::string> database_dirs = {
      "/home/steven/Data/place_recognition/grid_features/",
      "/home/steven/Data/place_recognition/features/"};
  std::vector<std::string> query_dirs = {
      "/home/steven/Data/place_recognition/features/"
      // "/home/steven/Data/place_recognition/grid_features/"
  };

  std::string map_path = "/home/steven/Data/place_recognition/global_map.pcd";

  auto sc_manager = std::make_unique<SCManager>();
  auto simulator = std::make_unique<LidarSimulator>();

  if (!simulator->load_map(map_path, 0.05)) {
    return -1;
  }

  // 辅助 lambda：从指定目录加载 KeyFrames
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

      int id = 0;
      bool is_360 =
          false; // [修改] 现在 grid_features 也是 180 度快照，不按 360 度处理
      try {
        id = std::stoi(fs::path(pcd_path).stem().string());
        // 即使 id >= 100000，用户现在也要求按 180 度 FOV 处理
      } catch (...) {
      }
      kfs.push_back({id, pcd_path, pose, is_360});
    }
    return kfs;
  };

  std::cout << "[Main] Loading Database Frames..." << std::endl;
  std::vector<KeyFrame> database_keyframes_all = load_from_dirs(database_dirs);
  std::vector<KeyFrame> database_keyframes;
  for (const auto &kf : database_keyframes_all) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(
        new pcl::PointCloud<pcl::PointXYZ>);
    if (pcl::io::loadPCDFile(kf.pcd_path, *cloud) != -1) {
      filterOutliers(cloud);
      sc_manager->setUseFovMask(!kf.is_360_fov);
      if (sc_manager->makeAndSaveScancontextAndKeys(*cloud)) {
        database_keyframes.push_back(kf);
      }
    }
  }

  std::cout << "[Main] Loading Query Poses..." << std::endl;
  std::vector<KeyFrame> query_keyframes = load_from_dirs(query_dirs);

  std::cout << "[Main] Database built with " << database_keyframes.size()
            << " frames." << std::endl;
  std::cout << "[Main] Query set contains " << query_keyframes.size()
            << " poses." << std::endl;

  if (database_keyframes.empty() || query_keyframes.empty()) {
    std::cerr << "[Main] Error: Missing database or query frames!" << std::endl;
    return -1;
  }

  int correct_matches = 0;
  int incorrect_matches = 0;
  int not_found = 0;
  int total_tests = 0;

  std::string output_dir = "/home/steven/Data/place_recognition/matchs/";
  fs::create_directories(output_dir);

  std::cout << "\n[Main] Starting Simulation & Verification..." << std::endl;
  std::cout << "---------------------------------------------------------------"
               "---------------------------"
            << std::endl;
  std::cout << "| " << std::setw(6) << "GT ID"
            << " | " << std::setw(8) << "Match ID"
            << " | " << std::setw(12) << "Est Yaw(deg)"
            << " | " << std::setw(12) << "Ang Err(deg)"
            << " | " << std::setw(8) << "Dist(m)"
            << " | " << std::setw(6) << "Status"
            << " |" << std::endl;
  std::cout << "---------------------------------------------------------------"
               "---------------------------"
            << std::endl;

  for (const auto &kf : query_keyframes) {
    auto sim_pose = matrixToPose(kf.pose);
    auto sim_cloud = simulator->simulate_scan(sim_pose);

    if (sim_cloud->empty())
      continue;

    filterOutliers(sim_cloud);

    // 对于在线仿真的点云，我们也需要确定其 FOV 掩码策略。
    // 这里假设仿真的点云（Query）默认使用 FOV 掩码（非360度）
    sc_manager->setUseFovMask(true);

    auto result = sc_manager->detectLoopClosureID(sim_cloud);

    // 1. Query SC
    sc_manager->setUseFovMask(true);
    Eigen::MatrixXd q_sc = sc_manager->makeScancontext(*sim_cloud);
    cv::Mat query_img = sc_manager->getScanContextVisual(q_sc);

    // 2. True SC (Ground Truth)
    // [修复] 直接从 query 帧对应的路径加载 PCD 作为 GT
    // 可视化，这样即使数据库只包含 grid_features 也能正常显示真值
    cv::Mat true_img = cv::Mat::zeros(query_img.rows, query_img.cols, CV_8UC3);
    pcl::PointCloud<pcl::PointXYZ>::Ptr true_cloud(
        new pcl::PointCloud<pcl::PointXYZ>);

    if (fs::exists(kf.pcd_path)) {
      if (pcl::io::loadPCDFile(kf.pcd_path, *true_cloud) != -1) {
        filterOutliers(true_cloud);
        sc_manager->setUseFovMask(!kf.is_360_fov);
        Eigen::MatrixXd t_sc = sc_manager->makeScancontext(*true_cloud);
        true_img = sc_manager->getScanContextVisual(t_sc);
      }
    } else {
      // 备选方案：如果 kf.pcd_path 不存在，尝试在数据库中寻找 ID 相同的作为 GT
      // 显示
      auto it = std::find_if(
          database_keyframes.begin(), database_keyframes.end(),
          [&](const KeyFrame &db_kf) { return db_kf.id == kf.id; });

      if (it != database_keyframes.end()) {
        if (pcl::io::loadPCDFile(it->pcd_path, *true_cloud) != -1) {
          filterOutliers(true_cloud);
          sc_manager->setUseFovMask(!it->is_360_fov);
          Eigen::MatrixXd t_sc = sc_manager->makeScancontext(*true_cloud);
          true_img = sc_manager->getScanContextVisual(t_sc);
        }
      }
    }

    // 3. Match SC (If found)
    cv::Mat match_img = cv::Mat::zeros(query_img.rows, query_img.cols, CV_8UC3);
    pcl::PointCloud<pcl::PointXYZ>::Ptr match_cloud(
        new pcl::PointCloud<pcl::PointXYZ>);

    match_cloud->clear();

    int match_idx = result.first;
    if (match_idx != -1 && match_idx < (int)database_keyframes.size()) {
      // 获取匹配到的帧是否为 360 度
      sc_manager->setUseFovMask(!database_keyframes[match_idx].is_360_fov);
      match_img = sc_manager->getScanContextVisual(
          sc_manager->polarcontexts_[match_idx]);
      if (pcl::io::loadPCDFile(database_keyframes[match_idx].pcd_path,
                               *match_cloud) == -1) {
        std::cerr << "Failed to load match PCD: "
                  << database_keyframes[match_idx].pcd_path << std::endl;
      }
      filterOutliers(match_cloud);
    }

    cv::Mat separator = cv::Mat::ones(query_img.rows, 2, CV_8UC3) * 255;
    cv::Mat display_img;
    std::vector<cv::Mat> imgs = {query_img, separator, true_img, separator,
                                 match_img};
    cv::hconcat(imgs, display_img);

    int offset1 = query_img.cols + separator.cols;
    int offset2 = offset1 + true_img.cols + separator.cols;

    cv::putText(display_img, "Query", cv::Point(10, 20),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
    cv::putText(display_img, "GroundTruth", cv::Point(offset1 + 10, 20),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
    cv::putText(display_img,
                "Match (Idx: " + std::to_string(result.first) + ")",
                cv::Point(offset2 + 10, 20), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                cv::Scalar(255, 255, 255), 1);

    cv::imshow("ScanContext (Query | True | Match)", display_img);
    cv::waitKey(1);

    if (!display_img.empty()) {
      std::string img_save_name = output_dir + std::to_string(kf.id) + ".jpg";
      cv::imwrite(img_save_name, display_img);
    }

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr combined_cloud(
        new pcl::PointCloud<pcl::PointXYZRGB>);

    auto add_colored = [&](pcl::PointCloud<pcl::PointXYZ>::Ptr src, uint8_t r,
                           uint8_t g, uint8_t b) {
      for (const auto &pt : src->points) {
        pcl::PointXYZRGB p;
        p.x = pt.x;
        p.y = pt.y;
        p.z = pt.z;
        p.r = r;
        p.g = g;
        p.b = b;
        combined_cloud->push_back(p);
      }
    };

    // Query: Green (0, 255, 0)
    add_colored(sim_cloud, 0, 255, 0);
    // GroundTruth: Red (255, 0, 0)
    add_colored(true_cloud, 255, 0, 0);
    // Match: Blue (0, 0, 255)
    if (!match_cloud->empty()) {
      // [要求] 根据估计出的旋转角（result.second）将匹配帧旋转对齐到 query 帧
      pcl::PointCloud<pcl::PointXYZ>::Ptr aligned_match(
          new pcl::PointCloud<pcl::PointXYZ>);
      Eigen::Affine3f transform = Eigen::Affine3f::Identity();
      transform.rotate(
          Eigen::AngleAxisf(result.second, Eigen::Vector3f::UnitZ()));
      pcl::transformPointCloud(*match_cloud, *aligned_match, transform);

      add_colored(aligned_match, 0, 0, 255);
    }

    if (!combined_cloud->empty()) {
      std::string save_name = output_dir + std::to_string(kf.id) + ".pcd";
      pcl::io::savePCDFileBinary(save_name, *combined_cloud);
    }

    // --- 状态统计 ---
    std::string s_out = "FAIL";
    double dist = -1.0;
    double angle_error = -1.0;

    if (result.first == -1) {
      s_out = "N/A";
      not_found++;
    } else {
      Eigen::Vector3d gt_pos = kf.pose.block<3, 1>(0, 3);
      Eigen::Vector3d match_pos =
          database_keyframes[result.first].pose.block<3, 1>(0, 3);
      dist = (gt_pos - match_pos).norm();

      // [要求] 计算夹角误差 (角度)
      double query_yaw = sim_pose.yaw;
      double match_yaw =
          matrixToPose(database_keyframes[result.first].pose).yaw;
      double gt_yaw_diff = normalizeAngle(match_yaw - query_yaw);
      double est_yaw_diff = result.second;
      angle_error =
          std::abs(normalizeAngle(est_yaw_diff - gt_yaw_diff)) * 180.0 / M_PI;

      // [要求] 夹角误差 < 30 度 且 距离 <= 7.0m 视为 OK
      bool is_match = (dist <= 7.0 && angle_error < 30.0);
      if (is_match) {
        s_out = "OK";
        correct_matches++;
      } else {
        incorrect_matches++;
      }
    }
    total_tests++;

    std::cout << "| " << std::setw(6) << kf.id << " | " << std::setw(8)
              << result.first << " | " << std::setw(12) << std::fixed
              << std::setprecision(2) << (result.second * 180.0 / M_PI)
              << " | ";

    if (result.first == -1) {
      std::cout << std::setw(12) << "---"
                << " | " << std::setw(8) << "---";
    } else {
      std::cout << std::setw(12) << std::fixed << std::setprecision(2)
                << angle_error << " | " << std::setw(8) << std::fixed
                << std::setprecision(2) << dist;
    }

    std::cout << " | " << std::setw(6) << s_out << " |" << std::endl;

    // [新增] FAILED 状态下的复盘逻辑
    if (s_out == "FAIL") {
      std::cout << "    >>> [REVIEW] Query ID " << kf.id << " failed."
                << std::endl;

      // 1. 获取当前 Query 的描述子
      sc_manager->setUseFovMask(true);
      Eigen::MatrixXd q_sc = sc_manager->makeScancontext(*sim_cloud);

      // 2. 检查真实帧 (GT) 在数据库中的分数
      auto it_gt = std::find_if(
          database_keyframes.begin(), database_keyframes.end(),
          [&](const KeyFrame &db_kf) { return db_kf.id == kf.id; });
      if (it_gt != database_keyframes.end()) {
        int gt_db_idx = std::distance(database_keyframes.begin(), it_gt);
        Eigen::MatrixXd gt_sc = sc_manager->polarcontexts_[gt_db_idx];
        auto gt_score = sc_manager->distanceBtnScanContext(q_sc, gt_sc);
        std::cout << "        -> GroundTruth (ID " << it_gt->id
                  << "): SC Dist = " << std::fixed << std::setprecision(4)
                  << gt_score.first << ", SC Yaw Align = "
                  << (gt_score.second * sc_manager->PC_UNIT_SECTORANGLE)
                  << " deg" << std::endl;
      } else {
        std::cout << "        -> GroundTruth: ID " << kf.id
                  << " not found in database." << std::endl;
      }

      // 3. 检查匹配结果的分数
      if (result.first != -1) {
        Eigen::MatrixXd match_sc = sc_manager->polarcontexts_[result.first];
        auto match_score = sc_manager->distanceBtnScanContext(q_sc, match_sc);
        std::cout << "        -> Erroneous Match (ID "
                  << database_keyframes[result.first].id
                  << "): SC Dist = " << std::fixed << std::setprecision(4)
                  << match_score.first << ", SC Yaw Align = "
                  << (match_score.second * sc_manager->PC_UNIT_SECTORANGLE)
                  << " deg" << std::endl;
      }

      // [新增] BEV Debugger 可视化
      pcl::PointCloud<pcl::PointXYZ>::Ptr debug_query = sim_cloud;
      pcl::PointCloud<pcl::PointXYZ>::Ptr debug_gt(
          new pcl::PointCloud<pcl::PointXYZ>);
      if (fs::exists(kf.pcd_path)) {
        if (pcl::io::loadPCDFile(kf.pcd_path, *debug_gt) != -1) {
          filterOutliers(debug_gt);
        }
      }
      BEVDebugger debugger;
      debugger.WINDOW_SIZE = 1000;
      debugger.SC_NUM_SECTORS = sc_manager->PC_NUM_SECTOR;
      debugger.SC_NUM_RINGS = sc_manager->PC_NUM_RING;
      std::cout << "        -> [DEBUG] Showing BEV diff map..." << std::endl;
      debugger.showDifference(debug_query, debug_gt, 10.0f,
                              "BEV Debugger");

      std::cout << "    <<<" << std::endl;
    }
  }

  std::cout << "---------------------------------------------------------------"
               "---------------------------"
            << std::endl;
  std::cout << "Correct:   " << correct_matches << " / " << total_tests << " ("
            << (total_tests > 0 ? (100.0 * correct_matches / total_tests) : 0.0)
            << "%)" << std::endl;
  std::cout << "Incorrect: " << incorrect_matches << " / " << total_tests
            << std::endl;
  std::cout << "Not Found: " << not_found << " / " << total_tests << std::endl;

  return 0;
}