#pragma once

// OpenCV Headers
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

// Standard & PCL Headers
#include <Eigen/Dense>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <pcl/common/common.h>
#include <pcl/common/io.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <string>
#include <thread>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace fs = std::filesystem;
typedef pcl::PointXYZ PointType;

struct DescriptorSet {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  Eigen::VectorXd solid;
  Eigen::VectorXd iev;
  Eigen::MatrixXd range_mat;
  Eigen::MatrixXd angle_mat;

  DescriptorSet() {
    solid = Eigen::VectorXd::Zero(1);
    iev = Eigen::VectorXd::Zero(1);
    range_mat = Eigen::MatrixXd::Zero(1, 1);
    angle_mat = Eigen::MatrixXd::Zero(1, 1);
  }
};

struct MatchResult {
  int id = -1;
  double score = 0.0;
  double heading = 0.0;
  double iev_score = 0.0;
};

struct VisualizationData {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  DescriptorSet query;
  DescriptorSet candidate;
  int cand_id = -1;
  double score = 0.0;
  double iev_score = 0.0;
  bool valid = false;
};

class SOLiDModule {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  const float MOUNT_PITCH_DEG = 0.0f;
  const float LIDAR_FOV_U = 52.0f;
  const float LIDAR_FOV_D = -7.0f;
  const float FOV_u = LIDAR_FOV_U - MOUNT_PITCH_DEG;
  const float FOV_d = LIDAR_FOV_D - MOUNT_PITCH_DEG;

  const int NUM_ANGLE = 60;
  const int NUM_RANGE = 60;
  const int NUM_HEIGHT = 80;
  const float MAX_DISTANCE = 12.0f;
  const float VOXEL_SIZE = 0.2f;

private:
  std::vector<DescriptorSet> database_features_;
  std::vector<int> database_ids_;

  std::thread viz_thread_;
  std::mutex viz_mutex_;
  std::condition_variable viz_cv_;
  std::atomic<bool> viz_running_{false};
  bool has_new_data_ = false;
  VisualizationData viz_data_;

public:
  SOLiDModule() {
    viz_running_ = true;
    viz_thread_ = std::thread(&SOLiDModule::visualization_thread, this);
  }

  ~SOLiDModule() {
    viz_running_ = false;
    viz_cv_.notify_all();
    if (viz_thread_.joinable())
      viz_thread_.join();
  }

  void preprocess(const pcl::PointCloud<PointType>::Ptr &scan_raw,
                  pcl::PointCloud<PointType>::Ptr &scan_out) {
    if (!scan_raw || scan_raw->empty())
      return;
    pcl::PointCloud<PointType>::Ptr temp(new pcl::PointCloud<PointType>());
    temp->reserve(scan_raw->size());
    for (const auto &pt : scan_raw->points) {
      if (std::isfinite(pt.x) && std::isfinite(pt.y) && std::isfinite(pt.z)) {
        if (pt.z > -0.45f && pt.z < 3.5f)
          temp->points.push_back(pt);
      }
    }
    pcl::VoxelGrid<PointType> voxel_grid;
    voxel_grid.setInputCloud(temp);
    voxel_grid.setLeafSize(VOXEL_SIZE, VOXEL_SIZE, VOXEL_SIZE);
    voxel_grid.filter(*scan_out);
  }

  void insert(const pcl::PointCloud<PointType>::Ptr &scan_raw, int id) {
    pcl::PointCloud<PointType>::Ptr processed(new pcl::PointCloud<PointType>());
    preprocess(scan_raw, processed);
    DescriptorSet ds = makeSolid(*processed);
    database_features_.push_back(ds);
    database_ids_.push_back(id);
  }

  MatchResult query(const pcl::PointCloud<PointType>::Ptr &scan_raw) {
    MatchResult result;
    if (database_features_.empty())
      return result;

    pcl::PointCloud<PointType>::Ptr processed(new pcl::PointCloud<PointType>());
    preprocess(scan_raw, processed);
    DescriptorSet query_set = makeSolid(*processed);

    double best_score = -1.0;
    int best_idx = -1;

    for (size_t i = 0; i < database_features_.size(); ++i) {
      double geo_score =
          loop_detection(query_set.solid, database_features_[i].solid);
      if (geo_score > best_score) {
        best_score = geo_score;
        best_idx = i;
      }
    }

    if (best_idx != -1) {
      double iev_val =
          compare_iev(query_set.iev, database_features_[best_idx].iev);
      result.iev_score = iev_val;
      // 简单逻辑：只返回最高分，哪怕很低，由外部判断阈值
      result.id = database_ids_[best_idx];
      result.score = best_score;

      {
        std::lock_guard<std::mutex> lock(viz_mutex_);
        viz_data_.query = query_set;
        viz_data_.candidate = database_features_[best_idx];
        viz_data_.cand_id = database_ids_[best_idx];
        viz_data_.score = best_score;
        viz_data_.iev_score = iev_val;
        viz_data_.valid = true;
        has_new_data_ = true;
      }
      viz_cv_.notify_one();
    }
    return result;
  }

private:
  void visualization_thread() {
    // 无需 namedWindow，imshow 会自动创建
    while (viz_running_) {
      VisualizationData current_viz;
      {
        std::unique_lock<std::mutex> lock(viz_mutex_);
        viz_cv_.wait(lock, [this] { return has_new_data_ || !viz_running_; });
        if (!viz_running_)
          break;
        current_viz = viz_data_;
        has_new_data_ = false;
      }
      if (current_viz.valid) {
        cv::Mat img = generate_debug_image(current_viz);
        if (!img.empty()) {
          cv::imshow("SOLiD Debugger", img);
          cv::waitKey(10);
        }
      }
    }
    cv::destroyAllWindows();
  }

  cv::Mat generate_debug_image(const VisualizationData &data) {
    cv::Mat q_mat = eigen2cv_heatmap(data.query.range_mat);
    cv::Mat c_mat = eigen2cv_heatmap(data.candidate.range_mat);
    cv::Mat q_iev = draw_curve(data.query.iev, "Query IEV");
    cv::Mat c_iev = draw_curve(data.candidate.iev, "Cand IEV");

    if (q_mat.empty() || c_mat.empty())
      return cv::Mat();

    cv::Mat top, bot, combined;
    cv::hconcat(q_mat, c_mat, top);
    cv::hconcat(q_iev, c_iev, bot);
    cv::vconcat(top, bot, combined);

    std::stringstream ss;
    ss << "Match ID: " << data.cand_id << " | Geo: " << std::fixed
       << std::setprecision(3) << (std::isnan(data.score) ? 0.0 : data.score)
       << " | IEV: " << (std::isnan(data.iev_score) ? 0.0 : data.iev_score);

    cv::rectangle(combined, cv::Point(0, 0), cv::Point(combined.cols, 40),
                  cv::Scalar(0, 0, 0), -1);
    cv::putText(combined, ss.str(), cv::Point(15, 30), cv::FONT_HERSHEY_SIMPLEX,
                0.7, cv::Scalar(0, 255, 0), 2);
    return combined;
  }

  cv::Mat eigen2cv_heatmap(const Eigen::MatrixXd &mat) {
    if (mat.rows() <= 1 || mat.cols() <= 1)
      return cv::Mat();
    double max_val = mat.maxCoeff();
    cv::Mat img(mat.cols(), mat.rows(), CV_8UC1);
    for (int r = 0; r < mat.rows(); ++r) {
      for (int c = 0; c < mat.cols(); ++c) {
        double val = (max_val > 1e-6) ? (mat(r, c) / max_val * 255.0) : 0.0;
        img.at<uchar>(mat.cols() - 1 - c, r) =
            static_cast<uchar>(std::clamp(val, 0.0, 255.0));
      }
    }
    cv::Mat color;
    cv::applyColorMap(img, color, cv::COLORMAP_JET);
    cv::resize(color, color, cv::Size(400, 300), 0, 0, cv::INTER_NEAREST);
    return color;
  }

  cv::Mat draw_curve(const Eigen::VectorXd &vec, std::string label) {
    cv::Mat img = cv::Mat::zeros(200, 400, CV_8UC3);
    if (vec.size() < 2)
      return img;
    double max_v = vec.maxCoeff();
    if (max_v < 1e-6)
      max_v = 1.0;
    for (int i = 0; i < vec.size() - 1; ++i) {
      cv::line(img,
               cv::Point(i * 400 / vec.size(), 200 - (vec(i) / max_v * 180)),
               cv::Point((i + 1) * 400 / vec.size(),
                         200 - (vec(i + 1) / max_v * 180)),
               cv::Scalar(0, 255, 255), 2);
    }
    cv::putText(img, label, cv::Point(10, 25), cv::FONT_HERSHEY_SIMPLEX, 0.6,
                cv::Scalar(255, 255, 255), 1);
    return img;
  }

  DescriptorSet makeSolid(pcl::PointCloud<PointType> &scan) {
    DescriptorSet ds;
    ds.range_mat = Eigen::MatrixXd::Zero(NUM_RANGE, NUM_HEIGHT);
    ds.angle_mat = Eigen::MatrixXd::Zero(NUM_ANGLE, NUM_HEIGHT);

    float g_a = 360.0f / NUM_ANGLE;
    float g_r = MAX_DISTANCE / NUM_RANGE;
    float g_h = (FOV_u - FOV_d) / NUM_HEIGHT;

    for (auto &pt : scan.points) {
      float d = std::sqrt(pt.x * pt.x + pt.y * pt.y);
      float ang = std::atan2(pt.y, pt.x) * 180.0f / M_PI;
      if (ang < 0)
        ang += 360.0f;
      float phi = std::atan2(pt.z, d) * 180.0f / M_PI;

      if (!std::isfinite(d) || !std::isfinite(ang) || !std::isfinite(phi))
        continue;

      int ir = std::clamp((int)(d / g_r), 0, NUM_RANGE - 1);
      int ia = std::clamp((int)(ang / g_a), 0, NUM_ANGLE - 1);
      int ih = std::clamp((int)((phi - FOV_d) / g_h), 0, NUM_HEIGHT - 1);

      ds.range_mat(ir, ih) += 1.0;
      ds.angle_mat(ia, ih) += 1.0;
    }

    Eigen::VectorXd iev = ds.range_mat.colwise().sum();
    double sum_iev = iev.sum();
    ds.iev = (sum_iev > 1e-6) ? (iev / sum_iev) : iev;

    double min_v = iev.minCoeff();
    double max_v = iev.maxCoeff();
    Eigen::VectorXd weight =
        (max_v - min_v > 1e-6)
            ? Eigen::VectorXd((iev.array() - min_v) / (max_v - min_v))
            : Eigen::VectorXd::Zero(NUM_HEIGHT);

    ds.solid = Eigen::VectorXd::Zero(NUM_RANGE + NUM_ANGLE);
    ds.solid << (ds.range_mat * weight), (ds.angle_mat * weight);

    return ds;
  }

  double loop_detection(const Eigen::VectorXd &q, const Eigen::VectorXd &c) {
    if (q.size() < NUM_RANGE || c.size() < NUM_RANGE)
      return 0.0;
    double dot = q.head(NUM_RANGE).dot(c.head(NUM_RANGE));
    double norm = q.head(NUM_RANGE).norm() * c.head(NUM_RANGE).norm();
    return (norm > 1e-6) ? (dot / norm) : 0.0;
  }

  double compare_iev(const Eigen::VectorXd &q, const Eigen::VectorXd &c) {
    if (q.size() != c.size() || q.size() == 0)
      return 0.0;
    double intersect = 0.0;
    for (int i = 0; i < q.size(); ++i)
      intersect += std::min(q(i), c(i));
    return intersect;
  }
};