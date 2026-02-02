// lidar_simulator.hpp

#pragma once

#include <Eigen/Geometry>
#include <array>
#include <cmath>
#include <iostream>
#include <memory>
#include <octomap/OcTree.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

class LidarSimulator {
private:
  // --- Mid-360 FOV Params ---
  const double LIVOX_FOV_MIN_RAD = -7.0 * M_PI / 180.0;
  const double LIVOX_FOV_MAX_RAD = 70.0 * M_PI / 180.0;

  // Installation: 45 degree tilt
  const double LIDAR_PITCH_OFFSET_RAD = 45.0 * M_PI / 180.0;

  const int DEFAULT_PTS_PER_FRAME = 260000;
  const double GOLDEN_ANGLE = M_PI * (3.0 - sqrt(5.0));

  std::unique_ptr<octomap::OcTree> octomap_;
  bool map_loaded_ = false;

public:
  struct Pose {
    double x, y, z, roll, pitch, yaw;
  };

  LidarSimulator() = default;

  bool load_map(const std::string &path, double resolution = 0.05) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(
        new pcl::PointCloud<pcl::PointXYZ>);
    if (pcl::io::loadPCDFile(path, *cloud) == -1) {
      std::cerr << "[Sim] Failed to load PCD: " << path << std::endl;
      return false;
    }

    std::cout << "[Sim] Building OctoMap from " << cloud->size() << " points..."
              << std::endl;
    octomap_ = std::make_unique<octomap::OcTree>(resolution);
    for (const auto &pt : cloud->points)
      octomap_->updateNode(pt.x, pt.y, pt.z, true);
    octomap_->updateInnerOccupancy();

    map_loaded_ = true;
    std::cout << "[Sim] Map loaded." << std::endl;
    return true;
  }

  void get_map_bounds(double &min_x, double &max_x, double &min_y, double &max_y, double &min_z, double &max_z) {
    if (!map_loaded_) return;
    octomap_->getMetricMin(min_x, min_y, min_z);
    octomap_->getMetricMax(max_x, max_y, max_z);
  }

  bool find_ground_height(double x, double y, double &ground_z, double search_z_start = 2.0) {
    if (!map_loaded_) return false;
    
    double min_x, min_y, min_z, max_x, max_y, max_z;
    octomap_->getMetricMin(min_x, min_y, min_z);
    octomap_->getMetricMax(max_x, max_y, max_z);

    octomap::point3d origin(x, y, search_z_start);
    octomap::point3d dir(0, 0, -1);
    octomap::point3d end;
    
    // Search downwards from search_z_start to min_z
    if (octomap_->castRay(origin, dir, end, true, search_z_start - min_z + 0.5)) {
        ground_z = end.z();
        return true;
    }
    return false;
  }

  // Input: Global Pose
  // Output: Point Cloud in Body/Lidar Frame
  pcl::PointCloud<pcl::PointXYZ>::Ptr simulate_scan(
      const Pose &pose, 
      int num_pts = 160000, 
      double fov_min_rad = -7.0 * M_PI / 180.0, 
      double fov_max_rad = 65.0 * M_PI / 180.0,
      double pitch_offset_rad = 45.0 * M_PI / 180.0) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr out_cloud(
        new pcl::PointCloud<pcl::PointXYZ>);
    if (!map_loaded_ || !octomap_)
      return out_cloud;

    out_cloud->reserve(num_pts);
    octomap::point3d origin(pose.x, pose.y, pose.z);

    const double z_min_sensor = sin(fov_min_rad);
    const double z_max_sensor = sin(fov_max_rad);
    const double z_range = z_max_sensor - z_min_sensor;

    // Precompute pose trig
    double cy = cos(pose.yaw), sy = sin(pose.yaw);
    double cp = cos(pose.pitch), sp = sin(pose.pitch);
    double cr = cos(pose.roll), sr = sin(pose.roll);

    for (int i = 0; i < num_pts; ++i) {
      double ratio = (double)i / num_pts;
      double dz_sensor = z_min_sensor + z_range * ratio;
      double r_xy_sensor = sqrt(1.0 - dz_sensor * dz_sensor);
      double theta_sensor = i * GOLDEN_ANGLE; // Simplified pattern

      double dx_sensor = r_xy_sensor * cos(theta_sensor);
      double dy_sensor = r_xy_sensor * sin(theta_sensor);

      // 1. Apply Installation Rotation (Ry(pitch_offset)) -> To Body Frame
      double r1_x = dx_sensor * cos(pitch_offset_rad) +
                    dz_sensor * sin(pitch_offset_rad);
      double r1_y = dy_sensor;
      double r1_z = -dx_sensor * sin(pitch_offset_rad) +
                    dz_sensor * cos(pitch_offset_rad);

      // 2. Apply Robot Pose (Body -> World) to get Ray Direction
      // Rotation sequence: Rz(y) * Ry(p) * Rx(r) * point
      double r2_y = r1_y * cr - r1_z * sr;
      double r2_z = r1_y * sr + r1_z * cr;

      double r3_x = r1_x * cp + r2_z * sp;
      double r3_z = -r1_x * sp + r2_z * cp;

      double dir_x = r3_x * cy - r2_y * sy;
      double dir_y = r3_x * sy + r2_y * cy;
      double dir_z = r3_z;

      octomap::point3d dir(dir_x, dir_y, dir_z);
      octomap::point3d end;

      // 3. Raycast
      if (octomap_->castRay(origin, dir, end, true, 15.0)) {
        // Hit point in Global Frame: (end.x, end.y, end.z)

        // 4. Transform Global Point back to Body Frame
        // This is what the Lidar sees relative to itself
        float rx = (float)end.x() - pose.x;
        float ry = (float)end.y() - pose.y;
        float rz = (float)end.z() - pose.z;

        // Inverse Rotation: R^T = Rx(-r) * Ry(-p) * Rz(-y)
        // 4.1 Inv Yaw
        double i_x1 = rx * cy + ry * sy;
        double i_y1 = -rx * sy + ry * cy;
        double i_z1 = rz;

        // 4.2 Inv Pitch
        double i_x2 = i_x1 * cp + i_z1 * sp;
        double i_z2 = -i_x1 * sp + i_z1 * cp;
        double i_y2 = i_y1;

        // 4.3 Inv Roll
        double i_x3 = i_x2;
        double i_y3 = i_y2 * cr - i_z2 * sr;
        double i_z3 = i_y2 * sr + i_z2 * cr;

        // Result is point in Body Frame (Lidar mount base)
        out_cloud->push_back(pcl::PointXYZ(i_x3, i_y3, i_z3));
      }
    }
    return out_cloud;
  }
};