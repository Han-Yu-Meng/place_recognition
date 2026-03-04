// lidar_simulator.hpp

#pragma once

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <array>
#include <cmath>
#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>
#include <vector>

// OctoMap
#include <octomap/OcTree.h>

// PCL Headers
#include <pcl/common/transforms.h>
#include <pcl/filters/crop_box.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/surface/convex_hull.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

enum class SimulationMethod {
  RAY_CASTING, // 基于 OctoMap 的光线投射
               // (模拟真实扫描线分布，精度高，速度受分辨率影响)
  HIDDEN_POINT_REMOVAL // 基于 PCL HPR 的隐点剔除
                       // (直接提取可见点，保留原始点云密度，速度快)
};

class LidarSimulator {
public:
  SimulationMethod method_ = SimulationMethod::RAY_CASTING;

  double LIVOX_FOV_MIN_RAD = -12.0 * M_PI / 180.0;
  double LIVOX_FOV_MAX_RAD = 65.0 * M_PI / 180.0;

  // 如果雷达是斜着装的，这里定义倾角
  const double LIDAR_PITCH_OFFSET_RAD = 45 * M_PI / 180.0;

  const double GOLDEN_ANGLE = M_PI * (3.0 - sqrt(5.0));

  // Data
  std::unique_ptr<octomap::OcTree> octomap_;       // 用于 Ray Casting
  pcl::PointCloud<pcl::PointXYZ>::Ptr global_map_; // 用于 HPR
  bool map_loaded_ = false;

  std::vector<Eigen::Vector3d> cached_scan_dirs_;
  int cached_num_pts_ = -1;

  std::vector<Eigen::Affine3d> T_body_lidars_;
  Eigen::Affine3d T_body_lidar_ = Eigen::Affine3d::Identity();

public:
  struct Pose {
    double x, y, z, roll, pitch, yaw;
  };

  LidarSimulator() : global_map_(new pcl::PointCloud<pcl::PointXYZ>) {
    T_body_lidar_.linear() =
        Eigen::AngleAxisd(45.0 * M_PI / 180.0, Eigen::Vector3d::UnitY())
            .toRotationMatrix();
  }

  bool load_config(const std::string &config_path) {
    cv::FileStorage fs(config_path, cv::FileStorage::READ);
    if (!fs.isOpened()) {
      std::cerr << "[Sim] Error: Could not open config file " << config_path
                << std::endl;
      return false;
    }

    if (!fs["lidar_fov_min"].empty()) {
      double fov_min_deg;
      fs["lidar_fov_min"] >> fov_min_deg;
      LIVOX_FOV_MIN_RAD = fov_min_deg * M_PI / 180.0;
    }

    if (!fs["lidar_fov_max"].empty()) {
      double fov_max_deg;
      fs["lidar_fov_max"] >> fov_max_deg;
      LIVOX_FOV_MAX_RAD = fov_max_deg * M_PI / 180.0;
    }

    // Load Extrinsic Matrices
    T_body_lidars_.clear();
    cv::FileNode lidars_node = fs["T_body_lidars"];
    if (lidars_node.isSeq()) {
      for (cv::FileNodeIterator it = lidars_node.begin(); it != lidars_node.end();
           ++it) {
        cv::FileNode node = *it;
        if (node.isMap()) {
          // Parse X, Y, Z, Roll, Pitch, Yaw (degrees)
          Pose p;
          p.x = (double)node["x"];
          p.y = (double)node["y"];
          p.z = (double)node["z"];
          p.roll = (double)node["roll"] * M_PI / 180.0;
          p.pitch = (double)node["pitch"] * M_PI / 180.0;
          p.yaw = (double)node["yaw"] * M_PI / 180.0;
          T_body_lidars_.push_back(getTransformFromPose(p));
        } else {
          cv::Mat Mi;
          node >> Mi;
          if (!Mi.empty() && Mi.rows == 4 && Mi.cols == 4) {
            Eigen::Affine3d Ti = Eigen::Affine3d::Identity();
            for (int r = 0; r < 4; r++)
              for (int c = 0; c < 4; c++)
                Ti(r, c) = Mi.at<double>(r, c);
            T_body_lidars_.push_back(Ti);
          }
        }
      }
    }

    if (T_body_lidars_.empty()) {
      // Load Single Extrinsic Matrix T_body_lidar (backward compatibility)
      cv::Mat T_cv;
      fs["T_body_lidar"] >> T_cv;
      if (!T_cv.empty() && T_cv.rows == 4 && T_cv.cols == 4) {
        for (int r = 0; r < 4; r++)
          for (int c = 0; c < 4; c++)
            T_body_lidar_(r, c) = T_cv.at<double>(r, c);
      }
      T_body_lidars_.push_back(T_body_lidar_);
    }

    // Invalidate cache
    cached_num_pts_ = -1;

    std::cout << "[Sim] Config loaded from " << config_path << std::endl;
    std::cout << "[Sim] FOV: [" << LIVOX_FOV_MIN_RAD * 180.0 / M_PI << ", "
              << LIVOX_FOV_MAX_RAD * 180.0 / M_PI << "] deg" << std::endl;
    std::cout << "[Sim] Num Lidars: " << T_body_lidars_.size() << std::endl;

    return true;
  }

  bool load_map(const std::string &path, double resolution = 0.05) {
    if (pcl::io::loadPCDFile(path, *global_map_) == -1) {
      std::cerr << "[Sim] Failed to load PCD: " << path << std::endl;
      return false;
    }

    std::cout << "[Sim] Map loaded with " << global_map_->size() << " points."
              << std::endl;
    std::cout << "[Sim] Building OctoMap (res=" << resolution << ")..."
              << std::endl;

    // 构建 OctoMap 用于 RayCasting
    octomap_ = std::make_unique<octomap::OcTree>(resolution);
    for (const auto &pt : global_map_->points)
      octomap_->updateNode(pt.x, pt.y, pt.z, true);
    octomap_->updateInnerOccupancy();

    map_loaded_ = true;
    return true;
  }

  void get_map_bounds(double &min_x, double &max_x, double &min_y,
                      double &max_y, double &min_z, double &max_z) {
    if (!map_loaded_)
      return;
    octomap_->getMetricMin(min_x, min_y, min_z);
    octomap_->getMetricMax(max_x, max_y, max_z);
  }

  /**
   * @brief 在指定 (x, y) 坐标附近寻找地面高度
   * @param x 目标 X 坐标
   * @param y 目标 Y 坐标
   * @param ground_z 输出搜索到的高度值
   * @param search_radius 搜索半径 (默认 2.0m)
   * @return true 如果找到有效高度，否则 false
   */
  bool find_ground_height(double x, double y, double &ground_z,
                          double search_radius = 2.0) {
    if (!map_loaded_ || global_map_->empty())
      return false;

    // 1. 设置裁剪框，只看目标点附近的垂直柱状区域
    pcl::CropBox<pcl::PointXYZ> crop;
    crop.setInputCloud(global_map_);

    // 设置裁剪范围：水平方向为 search_radius，垂直方向覆盖整个地图范围
    double min_x_m, max_x_m, min_y_m, max_y_m, min_z_m, max_z_m;
    get_map_bounds(min_x_m, max_x_m, min_y_m, max_y_m, min_z_m, max_z_m);

    crop.setMin(Eigen::Vector4f(x - search_radius, y - search_radius,
                                min_z_m - 1.0, 1.0));
    crop.setMax(Eigen::Vector4f(x + search_radius, y + search_radius,
                                max_z_m + 1.0, 1.0));

    std::vector<int> indices;
    crop.filter(indices);

    if (indices.empty())
      return false;

    float min_z = 1e6;
    bool found = false;

    for (int idx : indices) {
      float z = global_map_->points[idx].z;
      if (z < min_z) {
        min_z = z;
        found = true;
      }
    }

    if (found) {
      ground_z = min_z;
      return true;
    }
    return false;
  }

  // 对外统一接口
  pcl::PointCloud<pcl::PointXYZ>::Ptr
  simulate_scan(const Pose &pose,
                int num_pts = 100000 // 仅对 RayCasting 有效
  ) {
    if (!map_loaded_)
      return std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();

    pcl::PointCloud<pcl::PointXYZ>::Ptr combined_cloud(
        new pcl::PointCloud<pcl::PointXYZ>);

    for (const auto &T : T_body_lidars_) {
      // Temporarily set T_body_lidar_ for internal scan methods
      T_body_lidar_ = T;

      pcl::PointCloud<pcl::PointXYZ>::Ptr scan;
      if (method_ == SimulationMethod::RAY_CASTING) {
        // Divide num_pts among lidars to maintain similar density
        scan = simulate_scan_raycast(pose, num_pts / T_body_lidars_.size());
      } else {
        scan = simulate_scan_hpr(pose, 100);
      }
      *combined_cloud += *scan;
    }

    return combined_cloud;
  }

private:
  // 手写实现隐点剔除 (Hidden Point Removal)
  pcl::PointIndices::Ptr
  hiddenPointRemoval(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
                     pcl::PointXYZ viewpoint, double param_R) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr inverted_cloud(
        new pcl::PointCloud<pcl::PointXYZ>);
    inverted_cloud->points.reserve(cloud->size() + 1);

    // 1. 球面翻转 (Spherical Inversion)
    for (const auto &p : cloud->points) {
      pcl::PointXYZ p_rel;
      p_rel.x = p.x - viewpoint.x;
      p_rel.y = p.y - viewpoint.y;
      p_rel.z = p.z - viewpoint.z;

      double dist =
          std::sqrt(p_rel.x * p_rel.x + p_rel.y * p_rel.y + p_rel.z * p_rel.z);
      if (dist < 1e-6)
        dist = 1e-6;

      // 翻转公式: p' = p + 2*(R - dist) * (p/dist)
      pcl::PointXYZ p_inv;
      double factor = 2.0 * (param_R - dist) / dist;
      p_inv.x = p_rel.x + factor * p_rel.x;
      p_inv.y = p_rel.y + factor * p_rel.y;
      p_inv.z = p_rel.z + factor * p_rel.z;
      inverted_cloud->push_back(p_inv);
    }

    // 将视点（局部原点）加入点集用于计算凸包
    pcl::PointXYZ origin(0, 0, 0);
    inverted_cloud->push_back(origin);

    // 2. 计算凸包 (Convex Hull)
    pcl::ConvexHull<pcl::PointXYZ> hull;
    hull.setInputCloud(inverted_cloud);
    hull.setDimension(3);

    pcl::PointCloud<pcl::PointXYZ> hull_points;
    std::vector<pcl::Vertices> polygons;
    hull.reconstruct(hull_points, polygons);

    // 3. 提取可见点索引
    pcl::PointIndices::Ptr visible_indices(new pcl::PointIndices);
    hull.getHullPointIndices(*visible_indices);

    // 过滤掉视点索引
    auto it = std::find(visible_indices->indices.begin(),
                        visible_indices->indices.end(), (int)cloud->size());
    if (it != visible_indices->indices.end()) {
      visible_indices->indices.erase(it);
    }

    return visible_indices;
  }

  // --- 策略 A: Ray Casting (模拟特定扫描图案) ---
  pcl::PointCloud<pcl::PointXYZ>::Ptr simulate_scan_raycast(const Pose &pose,
                                                            int num_pts) {
    // 1. Update Cache if needed
    if (num_pts != cached_num_pts_) {
      cached_scan_dirs_.resize(num_pts);

      const double fov_min = LIVOX_FOV_MIN_RAD;
      const double fov_max = LIVOX_FOV_MAX_RAD;
      const double z_min_sensor = sin(fov_min);
      const double z_max_sensor = sin(fov_max);
      const double z_range = z_max_sensor - z_min_sensor;

      for (int i = 0; i < num_pts; ++i) {
        double ratio = (double)i / num_pts;
        double dz = z_min_sensor + z_range * ratio;
        double r_xy = sqrt(1.0 - dz * dz);
        double theta = i * GOLDEN_ANGLE;
        cached_scan_dirs_[i] =
            Eigen::Vector3d(r_xy * cos(theta), r_xy * sin(theta), dz);
      }
      cached_num_pts_ = num_pts;
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr out_cloud(
        new pcl::PointCloud<pcl::PointXYZ>);
    out_cloud->reserve(num_pts);

    Eigen::Affine3d T_world_body = getTransformFromPose(pose);

    Eigen::Affine3d T_world_sensor = T_world_body * T_body_lidar_;

    Eigen::Vector3d origin_eig = T_world_sensor.translation();
    octomap::point3d origin(origin_eig.x(), origin_eig.y(), origin_eig.z());

    Eigen::Matrix3d R_world_sensor = T_world_sensor.rotation();

    Eigen::Affine3d T_body_world = T_world_body.inverse();

    std::vector<pcl::PointXYZ> points_storage(num_pts);
    std::vector<bool> point_valid(num_pts, false);

#pragma omp parallel for
    for (int i = 0; i < num_pts; ++i) {
      // Transform direction: Sensor -> World
      Eigen::Vector3d dir_world_eig = R_world_sensor * cached_scan_dirs_[i];
      octomap::point3d dir(dir_world_eig.x(), dir_world_eig.y(),
                           dir_world_eig.z());

      octomap::point3d end;
      // castRay is thread-safe for const octree
      if (octomap_->castRay(origin, dir, end, true, 50.0)) {
        // Convert World -> Body
        Eigen::Vector3d pt_world(end.x(), end.y(), end.z());
        Eigen::Vector3d pt_body = T_body_world * pt_world;

        points_storage[i] =
            pcl::PointXYZ(pt_body.x(), pt_body.y(), pt_body.z());
        point_valid[i] = true;
      }
    }

    // 4. Gather results
    for (int i = 0; i < num_pts; ++i) {
      if (point_valid[i]) {
        out_cloud->push_back(points_storage[i]);
      }
    }

    return out_cloud;
  }

  // --- 策略 B: Hidden Point Removal (基于几何可见性) ---
  pcl::PointCloud<pcl::PointXYZ>::Ptr simulate_scan_hpr(const Pose &pose,
                                                        double range) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr out_cloud(
        new pcl::PointCloud<pcl::PointXYZ>);

    // Transform Body Pose
    Eigen::Affine3d T_world_body = getTransformFromPose(pose);
    Eigen::Affine3d T_world_sensor = T_world_body * T_body_lidar_;
    Eigen::Vector3d sensor_pos = T_world_sensor.translation();

    // 1. CropBox: 先把全局地图中，距离当前位置太远的点切掉，减少 HPR 计算量
    pcl::PointCloud<pcl::PointXYZ>::Ptr local_area(
        new pcl::PointCloud<pcl::PointXYZ>);
    pcl::CropBox<pcl::PointXYZ> box_filter;
    box_filter.setMin(Eigen::Vector4f(sensor_pos.x() - range,
                                      sensor_pos.y() - range,
                                      sensor_pos.z() - range, 1.0));
    box_filter.setMax(Eigen::Vector4f(sensor_pos.x() + range,
                                      sensor_pos.y() + range,
                                      sensor_pos.z() + range, 1.0));
    box_filter.setInputCloud(global_map_);
    box_filter.filter(*local_area);

    if (local_area->empty())
      return out_cloud;

    // 2. Custom Hidden Point Removal
    pcl::PointXYZ viewpoint_pt(sensor_pos.x(), sensor_pos.y(), sensor_pos.z());
    pcl::PointIndices::Ptr visible_indices =
        hiddenPointRemoval(local_area, viewpoint_pt, range);

    // 提取可见点 (World Frame)
    pcl::PointCloud<pcl::PointXYZ>::Ptr hpr_cloud(
        new pcl::PointCloud<pcl::PointXYZ>);
    pcl::copyPointCloud(*local_area, visible_indices->indices, *hpr_cloud);

    // 3. 转换到 Body Frame 并应用 FOV 过滤
    Eigen::Affine3d T_body_world = T_world_body.inverse();
    Eigen::Affine3d T_sensor_body = T_body_lidar_.inverse();

    for (const auto &pt_world : hpr_cloud->points) {
      // World -> Body
      Eigen::Vector3d p_w(pt_world.x, pt_world.y, pt_world.z);
      Eigen::Vector3d p_b = T_body_world * p_w;

      // Body -> Sensor (为了检查 FOV)
      Eigen::Vector3d p_s = T_sensor_body * p_b;

      // 检查 Range
      double d = p_s.norm();
      if (d < 0.5 || d > range)
        continue;

      // 检查 FOV (Elevation angle)
      double sin_ele = p_s.z() / d;
      if (sin_ele >= sin(LIVOX_FOV_MIN_RAD) &&
          sin_ele <= sin(LIVOX_FOV_MAX_RAD)) {
        // 输出 Body Frame 下的点
        out_cloud->push_back(pcl::PointXYZ(p_b.x(), p_b.y(), p_b.z()));
      }
    }

    return out_cloud;
  }

  Eigen::Affine3d getTransformFromPose(const Pose &p) {
    Eigen::Affine3d T = Eigen::Affine3d::Identity();
    T.translation() << p.x, p.y, p.z;

    Eigen::Matrix3d R;
    R = Eigen::AngleAxisd(p.yaw, Eigen::Vector3d::UnitZ()) *
        Eigen::AngleAxisd(p.pitch, Eigen::Vector3d::UnitY()) *
        Eigen::AngleAxisd(p.roll, Eigen::Vector3d::UnitX());
    T.linear() = R;
    return T;
  }
};