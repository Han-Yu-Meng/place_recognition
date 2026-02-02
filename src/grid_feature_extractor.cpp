// grid_feature_extractor.cpp

#include "lidar_simulator.hpp"
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <opencv2/opencv.hpp>

namespace fs = std::filesystem;

int main(int argc, char **argv) {
  std::string map_path = "/home/steven/Data/place_recognition/global_map.pcd";
  std::string output_dir = "/home/steven/Data/place_recognition/grid_features/";

  if (!fs::exists(output_dir)) {
    fs::create_directories(output_dir);
  }

  LidarSimulator simulator;
  if (!simulator.load_map(map_path, 0.05)) {
    return -1;
  }

  double min_x, max_x, min_y, max_y, min_z, max_z;
  simulator.get_map_bounds(min_x, max_x, min_y, max_y, min_z, max_z);

  double step = 1;
  double robot_height = 0.325;

  std::cout << "[Grid] Starting extraction with bounds: "
            << "X: [" << min_x << ", " << max_x << "], "
            << "Y: [" << min_y << ", " << max_y << "]" << std::endl;

  int id_counter = 100000;
  std::vector<cv::Point2f> extracted_points;

  for (double x = min_x; x <= max_x; x += step) {
    for (double y = min_y; y <= max_y; y += step) {
      double ground_z;
      // Search from a bit above the likely ceiling or just use max_z - a bit
      if (!simulator.find_ground_height(x, y, ground_z, 2.0)) {
        continue;
      }
      if (ground_z + robot_height > 2) {
        // Too high off the ground, likely not a valid ground point
        continue;
      }

      bool position_added = false;
      // [修改] 在每个位置点模拟 8 个方向 (0, 45, 90, 135, 180, 225, 270, 315 度)
      for (int yaw_deg = 0; yaw_deg < 360; yaw_deg += 45) {
        LidarSimulator::Pose pose;
        pose.x = x;
        pose.y = y;
        // pose.z = ground_z + robot_height;
		pose.z = 0;
        pose.roll = 0;
        pose.pitch = 0;
        pose.yaw = yaw_deg * M_PI / 180.0;

        // 模拟原始 360 度扫描
        auto full_scan =
            simulator.simulate_scan(pose, 50000, -M_PI / 2.0, M_PI / 2.0, 0.0);

        // [要求] 裁剪为 180 度 FOV (保留前方点云: x > 0)
        pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_scan(
            new pcl::PointCloud<pcl::PointXYZ>);
        for (const auto &pt : full_scan->points) {
          if (pt.x > 0) {
            filtered_scan->push_back(pt);
          }
        }

        if (filtered_scan->points.size() < 1000) {
          continue;
        }

        if (!position_added) {
          extracted_points.push_back(cv::Point2f(x, y));
          position_added = true;
        }

        std::cout << "[Grid] Id " << id_counter << " Simulating at (" << x << ", "
                  << y << ", " << pose.z << ") Yaw: " << yaw_deg << " deg" << std::endl;

        std::stringstream ss;
        ss << std::setw(6) << std::setfill('0') << id_counter;
        std::string base_name = ss.str();
        std::string pcd_filename = output_dir + base_name + ".pcd";
        std::string odom_filename = output_dir + base_name + ".odom";

        pcl::io::savePCDFileBinary(pcd_filename, *filtered_scan);

        // Save odom file (保存完整的 4x4 变换矩阵)
        std::ofstream odom_file(odom_filename);
        odom_file << std::fixed << std::setprecision(6);
        
        Eigen::Matrix3d R;
        R = Eigen::AngleAxisd(pose.yaw, Eigen::Vector3d::UnitZ()) *
            Eigen::AngleAxisd(pose.pitch, Eigen::Vector3d::UnitY()) *
            Eigen::AngleAxisd(pose.roll, Eigen::Vector3d::UnitX());

        odom_file << R(0,0) << " " << R(0,1) << " " << R(0,2) << " " << pose.x << "\n";
        odom_file << R(1,0) << " " << R(1,1) << " " << R(1,2) << " " << pose.y << "\n";
        odom_file << R(2,0) << " " << R(2,1) << " " << R(2,2) << " " << pose.z << "\n";
        odom_file << 0.0 << " " << 0.0 << " " << 0.0 << " " << 1.0 << "\n";
        odom_file.close();

        id_counter++;
      }
    }
  }

  // [新增] 可视化已提取的点位
  if (!extracted_points.empty()) {
    int img_w = 800, img_h = 800;
    cv::Mat map_img = cv::Mat::zeros(img_h, img_w, CV_8UC3);
    
    float scale_x = (img_w - 100) / (max_x - min_x);
    float scale_y = (img_h - 100) / (max_y - min_y);
    float scale = std::min(scale_x, scale_y);

    auto to_img = [&](float x, float y) {
      int ix = 50 + (int)((x - min_x) * scale);
      int iy = img_h - 50 - (int)((y - min_y) * scale); // 翻转 Y 轴
      return cv::Point(ix, iy);
    };

    // 绘制地图边框
    cv::rectangle(map_img, to_img(min_x, min_y), to_img(max_x, max_y), cv::Scalar(100, 100, 100), 2);

    // 绘制点
    for (const auto& pt : extracted_points) {
      cv::circle(map_img, to_img(pt.x, pt.y), 3, cv::Scalar(0, 255, 0), -1);
    }

    std::string map_save_path = output_dir + "extraction_map.jpg";
    cv::imwrite(map_save_path, map_img);
    std::cout << "[Grid] Extraction map saved to: " << map_save_path << std::endl;
    cv::imshow("Extraction Plan", map_img);
    cv::waitKey(2000); // 显示 2 秒
  }

  std::cout << "[Grid] Extraction finished. Total points: " << extracted_points.size() << std::endl;
  return 0;
}
