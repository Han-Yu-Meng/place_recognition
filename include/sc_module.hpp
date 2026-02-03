#pragma once

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>
#include <vector>

// OpenCV
#include <opencv2/opencv.hpp>

// PCL
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

// 依赖库
#include "KDTreeVectorOfVectorsAdaptor.h"
#include "nanoflann.hpp"

using namespace Eigen;
using namespace nanoflann;

using SCPointType = pcl::PointXYZ;
using KeyMat = std::vector<std::vector<float>>;
using InvKeyTree = KDTreeVectorOfVectorsAdaptor<KeyMat, float>;

// --- 辅助工具函数 ---
inline float xy2theta(const float &_x, const float &_y) {
  // 使用 atan2 替代手动象限判断，更安全且能处理 x=0 的情况
  float angle = atan2(_y, _x) * 180.0 / M_PI;
  if (angle < 0) angle += 360.0;
  return angle;
}

inline MatrixXd circshift(MatrixXd &_mat, int _num_shift) {
  if (_num_shift == 0)
    return _mat;
  MatrixXd shifted_mat = MatrixXd::Zero(_mat.rows(), _mat.cols());
  for (int col_idx = 0; col_idx < _mat.cols(); col_idx++) {
    int new_location = (col_idx + _num_shift) % _mat.cols();
    shifted_mat.col(new_location) = _mat.col(col_idx);
  }
  return shifted_mat;
}

inline std::vector<float> eig2stdvec(MatrixXd _eigmat) {
  std::vector<float> vec(_eigmat.data(), _eigmat.data() + _eigmat.size());
  return vec;
}

class SCManager {
public:
  // --- 参数配置 ---
  const double LIDAR_HEIGHT = 0.0;
  const double LIDAR_CEILING_HEIGHT = 2.0;
  const double LIDAR_FLOOR_HEIGHT = -0.3;

  // Scan Context 尺寸
  const int PC_NUM_RING = 15;
  const int PC_NUM_SECTOR = 40;
  const double PC_MAX_RADIUS = 15.0; // 建议设为 10-20m 适合室内/局部调试
  const double PC_UNIT_SECTORANGLE = 360.0 / double(PC_NUM_SECTOR);

  // 匹配参数
  const int NUM_EXCLUDE_RECENT = 30;
  const int NUM_CANDIDATES_FROM_TREE = 10;
  const double SEARCH_RATIO = 0.1;
  const double SC_DIST_THRES = 0.6;

  // 可视化参数
  const int VIZ_WIN_SIZE = 800; // 每张子图的大小 (调高分辨率)

  // FOV 掩码
  bool use_fov_mask_ = true;
  void setUseFovMask(bool use) { use_fov_mask_ = use; }

  // --- 数据存储 ---
  std::vector<Eigen::MatrixXd> polarcontexts_;
  KeyMat polarcontext_invkeys_mat_;
  std::unique_ptr<InvKeyTree> polarcontext_tree_;

public:
  SCManager() = default;

  // 1. 生成 Scan Context
  Eigen::MatrixXd makeScancontext(pcl::PointCloud<SCPointType> &_scan_down) {
    const float NO_POINT = -1000.0;
    MatrixXd desc = NO_POINT * MatrixXd::Ones(PC_NUM_RING, PC_NUM_SECTOR);

    for (const auto &pt_raw : _scan_down.points) {
      float pz = pt_raw.z + LIDAR_HEIGHT;
      if (pz > LIDAR_CEILING_HEIGHT || pz < LIDAR_FLOOR_HEIGHT)
        continue;

      float r = sqrt(pt_raw.x * pt_raw.x + pt_raw.y * pt_raw.y);
      float theta = xy2theta(pt_raw.x, pt_raw.y);
      if (r > PC_MAX_RADIUS || r < 0.1)
        continue;

      int r_idx = std::min(PC_NUM_RING - 1,
                           int(floor((r / PC_MAX_RADIUS) * PC_NUM_RING)));
      int s_idx = std::min(PC_NUM_SECTOR - 1,
                           int(floor((theta / 360.0) * PC_NUM_SECTOR)));

      if (desc(r_idx, s_idx) < pz)
        desc(r_idx, s_idx) = pz;
    }

    for (int r = 0; r < desc.rows(); r++)
      for (int s = 0; s < desc.cols(); s++)
        if (desc(r, s) == NO_POINT)
          desc(r, s) = 0;

    if (use_fov_mask_)
      maskBlindSpot(desc);
    return desc;
  }

  void maskBlindSpot(Eigen::MatrixXd &desc) {
    int start = int(90.0 / PC_UNIT_SECTORANGLE);
    int end = int(270.0 / PC_UNIT_SECTORANGLE);
    for (int col = start; col < end; ++col)
      desc.col(col).setZero();
  }

  // 2. 数据库管理
  bool makeAndSaveScancontextAndKeys(pcl::PointCloud<SCPointType> &_scan_down) {
    Eigen::MatrixXd sc = makeScancontext(_scan_down);
    Eigen::MatrixXd ringkey = sc.rowwise().mean();

    polarcontexts_.push_back(sc);
    polarcontext_invkeys_mat_.push_back(eig2stdvec(ringkey));
    return true;
  }

  // 3. 核心可视化逻辑 (BEV + Grid)
  cv::Mat generateDebugView(const pcl::PointCloud<SCPointType>::Ptr &cloud,
                            const std::string &label) {
    cv::Mat view = cv::Mat::zeros(VIZ_WIN_SIZE, VIZ_WIN_SIZE, CV_8UC3);
    if (!cloud || cloud->empty()) {
      cv::putText(view, label + " (Empty)", cv::Point(20, 40), 1, 1.5,
                  cv::Scalar(0, 0, 255), 2);
      return view;
    }

    float scale = (VIZ_WIN_SIZE / 2.0f) / (float)PC_MAX_RADIUS;
    int center = VIZ_WIN_SIZE / 2;

    // A. 高度缓存
    cv::Mat height_grid =
        cv::Mat::ones(VIZ_WIN_SIZE, VIZ_WIN_SIZE, CV_32F) * -999.0f;
    for (const auto &pt : cloud->points) {
      int row = center - static_cast<int>(pt.x * scale);
      int col = center - static_cast<int>(pt.y * scale);
      if (row >= 0 && row < VIZ_WIN_SIZE && col >= 0 && col < VIZ_WIN_SIZE) {
        if (pt.z > height_grid.at<float>(row, col))
          height_grid.at<float>(row, col) = pt.z;
      }
    }

    // B. 高度图渲染 + 膨胀
    cv::Mat gray = cv::Mat::zeros(VIZ_WIN_SIZE, VIZ_WIN_SIZE, CV_8UC1);
    for (int r = 0; r < VIZ_WIN_SIZE; r++) {
      for (int c = 0; c < VIZ_WIN_SIZE; c++) {
        float z = height_grid.at<float>(r, c);
        if (z > -900.0f) {
          float norm = std::clamp(
              (z + 1.0f) / (float)(LIDAR_CEILING_HEIGHT + 1.0f), 0.0f, 1.0f);
          gray.at<uchar>(r, c) = static_cast<uchar>(norm * 255);
        }
      }
    }
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::dilate(gray, gray, kernel);
    cv::applyColorMap(gray, view, cv::COLORMAP_JET);

    // 背景置黑
    for (int r = 0; r < VIZ_WIN_SIZE; r++)
      for (int c = 0; c < VIZ_WIN_SIZE; c++)
        if (gray.at<uchar>(r, c) == 0)
          view.at<cv::Vec3b>(r, c) = cv::Vec3b(0, 0, 0);

    // C. 绘制网格
    cv::Scalar grid_color(80, 80, 80);
    for (int i = 1; i <= PC_NUM_RING; i++) {
      int r_px = static_cast<int>((PC_MAX_RADIUS / PC_NUM_RING) * i * scale);
      cv::circle(view, cv::Point(center, center), r_px, grid_color, 1,
                 cv::LINE_AA);
    }
    for (int i = 0; i < PC_NUM_SECTOR; i++) {
      float rad = (i * 360.0 / PC_NUM_SECTOR) * M_PI / 180.0;
      int u = center - static_cast<int>(PC_MAX_RADIUS * sin(rad) * scale);
      int v = center - static_cast<int>(PC_MAX_RADIUS * cos(rad) * scale);
      cv::line(view, cv::Point(center, center), cv::Point(u, v), grid_color, 1,
               cv::LINE_AA);
    }

    // D. 标签
    cv::rectangle(view, cv::Point(0, 0), cv::Point(VIZ_WIN_SIZE, 40),
                  cv::Scalar(0, 0, 0), -1);
    cv::putText(view, label, cv::Point(15, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8,
                cv::Scalar(255, 255, 255), 2);
    return view;
  }

  /**
   * @brief 将 ScanContext 矩阵转换为极坐标热力图 (与 BEV 视角一致)
   */
  cv::Mat getPolarSCVisual(const Eigen::MatrixXd &_sc, const std::string &label) {
    cv::Mat view = cv::Mat::zeros(VIZ_WIN_SIZE, VIZ_WIN_SIZE, CV_8UC3);
    float scale = (VIZ_WIN_SIZE / 2.0f) / (float)PC_MAX_RADIUS;
    int center = VIZ_WIN_SIZE / 2;

    cv::Mat gray = cv::Mat::zeros(VIZ_WIN_SIZE, VIZ_WIN_SIZE, CV_8UC1);

    for (int r = 0; r < VIZ_WIN_SIZE; r++) {
      for (int c = 0; c < VIZ_WIN_SIZE; c++) {
        float dx = (center - r) / scale; // 还原视角中的 x (对应图像行偏移)
        float dy = (center - c) / scale; // 还原视角中的 y (对应图像列偏移)
        float dist = std::sqrt(dx * dx + dy * dy);
        
        if (dist > 0 && dist < PC_MAX_RADIUS) {
          float theta = xy2theta(dx, dy);
          int ring_idx = std::max(0, std::min(PC_NUM_RING - 1, (int)(dist / (PC_MAX_RADIUS / PC_NUM_RING))));
          int sector_idx = std::max(0, std::min(PC_NUM_SECTOR - 1, (int)(theta / PC_UNIT_SECTORANGLE)));
          
          float val = _sc(ring_idx, sector_idx);
          gray.at<uchar>(r, c) = static_cast<uchar>(std::clamp(255.0 * val / LIDAR_CEILING_HEIGHT, 0.0, 255.0));
        }
      }
    }

    cv::applyColorMap(gray, view, cv::COLORMAP_JET);
    
    // 背景置黑
    for (int r = 0; r < VIZ_WIN_SIZE; r++)
      for (int c = 0; c < VIZ_WIN_SIZE; c++)
        if (gray.at<uchar>(r, c) == 0)
          view.at<cv::Vec3b>(r, c) = cv::Vec3b(0, 0, 0);

    // 绘制网格线
    cv::Scalar grid_color(100, 100, 100);
    for (int i = 1; i <= PC_NUM_RING; i++) {
      int r_px = static_cast<int>((PC_MAX_RADIUS / PC_NUM_RING) * i * scale);
      cv::circle(view, cv::Point(center, center), r_px, grid_color, 1, cv::LINE_AA);
    }
    for (int i = 0; i < PC_NUM_SECTOR; i++) {
      float rad = (i * 360.0 / PC_NUM_SECTOR) * M_PI / 180.0;
      int u = center - static_cast<int>(PC_MAX_RADIUS * sin(rad) * scale);
      int v = center - static_cast<int>(PC_MAX_RADIUS * cos(rad) * scale);
      cv::line(view, cv::Point(center, center), cv::Point(u, v), grid_color, 1, cv::LINE_AA);
    }

    // 添加标签
    cv::rectangle(view, cv::Point(0, 0), cv::Point(300, 50), cv::Scalar(0, 0, 0), -1);
    cv::putText(view, label, cv::Point(15, 35), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 2);
    return view;
  }

  /**
   * @brief 生成三联 BEV 视图 Mat
   */
  cv::Mat getCombinedBEVDebugView(const pcl::PointCloud<SCPointType>::Ptr &q_cloud,
                                  const pcl::PointCloud<SCPointType>::Ptr &gt_cloud,
                                  const pcl::PointCloud<SCPointType>::Ptr &m_cloud) {
    cv::Mat v1 = generateDebugView(q_cloud, "Query BEV");
    cv::Mat v2 = generateDebugView(gt_cloud, "GroundTruth BEV");
    cv::Mat v3 = generateDebugView(m_cloud, "Best Match BEV");

    cv::Mat combined;
    std::vector<cv::Mat> views = {v1, v2, v3};
    cv::hconcat(views, combined);

    cv::line(combined, cv::Point(VIZ_WIN_SIZE, 0), cv::Point(VIZ_WIN_SIZE, VIZ_WIN_SIZE), cv::Scalar(255, 255, 255), 2);
    cv::line(combined, cv::Point(VIZ_WIN_SIZE * 2, 0), cv::Point(VIZ_WIN_SIZE * 2, VIZ_WIN_SIZE), cv::Scalar(255, 255, 255), 2);
    return combined;
  }

  /**
   * @brief 生成三联 ScanContext 矩阵视图 (极坐标视角)
   */
  cv::Mat getCombinedSCDebugView(const pcl::PointCloud<SCPointType>::Ptr &q_cloud,
                                 const pcl::PointCloud<SCPointType>::Ptr &gt_cloud,
                                 const pcl::PointCloud<SCPointType>::Ptr &m_cloud) {
    // 为确保展示的是“原始旋转角”，我们关闭掩码重新提取或者直接提取
    bool prev_mask = use_fov_mask_;
    setUseFovMask(false); // 可视化 SC 时显示完整 360 度便于对比旋转

    cv::Mat s_q = getPolarSCVisual(makeScancontext(*q_cloud), "Query SC (Polar)");
    cv::Mat s_gt = getPolarSCVisual(makeScancontext(*gt_cloud), "GT SC (Polar)");
    cv::Mat s_m = getPolarSCVisual(makeScancontext(*m_cloud), "Match SC (Polar)");

    setUseFovMask(prev_mask);

    cv::Mat combined;
    std::vector<cv::Mat> views = {s_q, s_gt, s_m};
    cv::hconcat(views, combined);

    // 加入分隔线
    cv::line(combined, cv::Point(VIZ_WIN_SIZE, 0), cv::Point(VIZ_WIN_SIZE, VIZ_WIN_SIZE), cv::Scalar(255, 255, 255), 2);
    cv::line(combined, cv::Point(VIZ_WIN_SIZE * 2, 0), cv::Point(VIZ_WIN_SIZE * 2, VIZ_WIN_SIZE), cv::Scalar(255, 255, 255), 2);
    return combined;
  }

  /**
   * @brief 显示调试三联图 (Query | GT | Match)
   */
  void showTripletDebug(const pcl::PointCloud<SCPointType>::Ptr &q_cloud,
                        const pcl::PointCloud<SCPointType>::Ptr &gt_cloud,
                        const pcl::PointCloud<SCPointType>::Ptr &m_cloud,
                        const std::string &win_name = "Triplet Debugger") {

    cv::Mat bev_view = getCombinedBEVDebugView(q_cloud, gt_cloud, m_cloud);
    cv::Mat sc_view = getCombinedSCDebugView(q_cloud, gt_cloud, m_cloud);

    cv::namedWindow(win_name + " - BEV", cv::WINDOW_NORMAL);
    cv::imshow(win_name + " - BEV", bev_view);

    cv::namedWindow(win_name + " - SC", cv::WINDOW_NORMAL);
    cv::imshow(win_name + " - SC", sc_view);
    
    cv::waitKey(1);
  }

  // 4. 搜索逻辑 (简化)
  std::pair<int, float>
  detectLoopClosureID(pcl::PointCloud<SCPointType>::Ptr q_cloud) {
    if (polarcontext_invkeys_mat_.size() < NUM_EXCLUDE_RECENT)
      return {-1, 0.0f};

    Eigen::MatrixXd q_desc = makeScancontext(*q_cloud);

    // 简单线性搜索 (为了调试代码简洁，这里没用 Tree)
    int best_id = -1;
    double min_dist = 10.0;
    int best_shift = 0;

    for (size_t i = 0; i < polarcontexts_.size(); i++) {
      auto res = distanceBtnScanContext(q_desc, polarcontexts_[i]);
      if (res.first < min_dist) {
        min_dist = res.first;
        best_id = i;
        best_shift = res.second;
      }
    }

    if (min_dist > SC_DIST_THRES)
      return {-1, 0.0f};
    return {best_id, (float)(best_shift * PC_UNIT_SECTORANGLE * M_PI / 180.0)};
  }

  std::pair<double, int> distanceBtnScanContext(const MatrixXd &_sc1,
                                                const MatrixXd &_sc2) {
    int best_shift = 0;
    double min_dist = 10.0;

    int num_cols = _sc1.cols();
    int search_width = std::ceil(30.0 / PC_UNIT_SECTORANGLE); // +/- 30 degrees
    std::vector<int> shift_indices;
    shift_indices.push_back(0);
    for(int i = 1; i <= search_width; ++i) {
        shift_indices.push_back(i);
        shift_indices.push_back(num_cols - i);
    }

    // Precompute norms to avoid recomputing inside loop
    double norm_sc1 = _sc1.norm();
    double norm_sc2 = _sc2.norm(); // Norm is invariant to circular shift
    double dist_base = norm_sc1 * norm_sc2 + 1e-6;

    for (int s : shift_indices) {
      double dot_sum = 0.0;
      // Optimization: Compute dot product without full matrix allocation (circshift)
      for (int col_idx = 0; col_idx < num_cols; col_idx++) {
        int sc1_col = (col_idx + s) % num_cols;
        dot_sum += _sc1.col(sc1_col).dot(_sc2.col(col_idx));
      }

      double dist = 1.0 - (dot_sum / dist_base);
      if (dist < min_dist) {
        min_dist = dist;
        best_shift = s;
      }
    }
    return {min_dist, best_shift};
  }
};