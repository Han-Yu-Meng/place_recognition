// sc_module.hpp

#pragma once

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <opencv2/opencv.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <vector>

// 依赖库 (请确保你的环境中包含这些头文件)
#include "KDTreeVectorOfVectorsAdaptor.h"
#include "nanoflann.hpp"
#include "tictoc.h" // 如果没有这个文件，可以用简单的 chrono 替代

using namespace Eigen;
using namespace nanoflann;

// 类型定义
using SCPointType = pcl::PointXYZ; // 为了兼容仿真器，这里改用 PointXYZ
using KeyMat = std::vector<std::vector<float>>;
using InvKeyTree = KDTreeVectorOfVectorsAdaptor<KeyMat, float>;

// --- 辅助函数 ---
inline float xy2theta(const float &_x, const float &_y) {
  if (_x >= 0 & _y >= 0)
    return (180 / M_PI) * atan(_y / _x);
  if (_x < 0 & _y >= 0)
    return 180 - ((180 / M_PI) * atan(_y / (-_x)));
  if (_x < 0 & _y < 0)
    return 180 + ((180 / M_PI) * atan(_y / _x));
  if (_x >= 0 & _y < 0)
    return 360 - ((180 / M_PI) * atan((-_y) / _x));
  return 0;
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
  const double LIDAR_HEIGHT =
      0.0; // 这里的 Height 是指雷达安装高度补偿，如果输入点云已经是 Body
           // 系，设为 0
  const double LIDAR_CEILING_HEIGHT =
      2.7; // [新增] 天花板高度阈值，超过此高度的点被忽略

  // FOV 设置
  bool use_fov_mask_ = true;          // 是否启用盲区遮罩
  const double FOV_VALID_DEG = 180.0; // 有效 FOV 角度 (假设正前方对称分布)
  // 假设正前方是 0 度 (X轴)，有效范围是 [-90, 90]。
  // 在 xy2theta (0~360) 体系下，有效范围是 [0, 90] 和 [270, 360]。
  // 盲区范围是 [90, 270]。

  void setUseFovMask(bool use) { use_fov_mask_ = use; }
  bool getUseFovMask() const { return use_fov_mask_; }

  const int PC_NUM_RING = 30;
  const int PC_NUM_SECTOR = 80;
  const double PC_MAX_RADIUS = 20.0;
  const double PC_UNIT_SECTORANGLE = 360.0 / double(PC_NUM_SECTOR);
  const double PC_UNIT_RINGGAP = PC_MAX_RADIUS / double(PC_NUM_RING);

  const int NUM_EXCLUDE_RECENT = 30;
  const int NUM_CANDIDATES_FROM_TREE = 20;
  const double SEARCH_RATIO = 0.1;
  const double SC_DIST_THRES = 0.6;

  int tree_making_period_conter = 0;
  const int TREE_MAKING_PERIOD_ = 50;

  // --- 数据存储 ---
  std::vector<Eigen::MatrixXd> polarcontexts_;
  std::vector<Eigen::MatrixXd> polarcontext_invkeys_;
  std::vector<Eigen::MatrixXd> polarcontext_vkeys_;
  KeyMat polarcontext_invkeys_mat_;
  KeyMat polarcontext_invkeys_to_search_;
  std::unique_ptr<InvKeyTree> polarcontext_tree_;

public:
  SCManager() = default;

  // 核心函数：生成 SC，并应用天花板过滤
  Eigen::MatrixXd makeScancontext(pcl::PointCloud<SCPointType> &_scan_down) {
    int num_pts_scan_down = _scan_down.points.size();
    const float NO_POINT = -1000.0;
    MatrixXd desc = NO_POINT * MatrixXd::Ones(PC_NUM_RING, PC_NUM_SECTOR);

    SCPointType pt;
    float azim_angle, azim_range;
    int ring_idx, sctor_idx;

    for (int pt_idx = 0; pt_idx < num_pts_scan_down; pt_idx++) {
      pt.x = _scan_down.points[pt_idx].x;
      pt.y = _scan_down.points[pt_idx].y;
      pt.z = _scan_down.points[pt_idx].z + LIDAR_HEIGHT;

      // [新增] 天花板过滤
      if (pt.z > LIDAR_CEILING_HEIGHT)
        continue;

      azim_range = sqrt(pt.x * pt.x + pt.y * pt.y);
      azim_angle = xy2theta(pt.x, pt.y);

      if (azim_range > PC_MAX_RADIUS)
        continue;

      ring_idx = std::max(
          std::min(PC_NUM_RING,
                   int(ceil((azim_range / PC_MAX_RADIUS) * PC_NUM_RING))),
          1);
      sctor_idx =
          std::max(std::min(PC_NUM_SECTOR,
                            int(ceil((azim_angle / 360.0) * PC_NUM_SECTOR))),
                   1);

      if (desc(ring_idx - 1, sctor_idx - 1) < pt.z)
        desc(ring_idx - 1, sctor_idx - 1) = pt.z;
    }

    // Reset NO_POINT to 0
    for (int row_idx = 0; row_idx < desc.rows(); row_idx++)
      for (int col_idx = 0; col_idx < desc.cols(); col_idx++)
        if (desc(row_idx, col_idx) == NO_POINT)
          desc(row_idx, col_idx) = 0;

    // [新增] 应用盲区遮罩
    if (use_fov_mask_) {
      maskBlindSpot(desc);
    }

    // 环权重分配：增加远处权重，降低近处权重
    for (int ring_idx = 0; ring_idx < PC_NUM_RING; ring_idx++) {
      double weight = std::min(
          0.5 + 1.0 * (double(ring_idx) / double(PC_NUM_RING - 1)), 1.2);
      desc.row(ring_idx) *= weight;
    }

    return desc;
  }

  // [新增] 强制将盲区列置零
  void maskBlindSpot(Eigen::MatrixXd &desc) {
    // 假设有效 FOV 为 180 度，正对前方 X 轴
    // 盲区范围：90度 到 270度
    // 对应 Sector 索引：
    // 90度 / 6度 = 15
    // 270度 / 6度 = 45
    // 盲区索引区间: [15, 45)  (不包含45)

    int start_idx = int(90.0 / PC_UNIT_SECTORANGLE);
    int end_idx = int(270.0 / PC_UNIT_SECTORANGLE);

    for (int col = start_idx; col < end_idx; ++col) {
      desc.col(col).setZero();
    }
  }

  // [修改] 检查 ScanContext 是否过于稀疏，并返回结果
  bool isSCSparse(const Eigen::MatrixXd &desc, const std::string &type) {
    int total_cells = desc.rows() * desc.cols();
    int non_zero = (desc.array() != 0).count();
    float ratio = (float)non_zero / total_cells;
    
    // 如果启用掩码，全零列不计入分母以获得更准确的稀疏度评估
    if (use_fov_mask_) {
      int blind_start_idx = int(90.0 / PC_UNIT_SECTORANGLE);
      int blind_end_idx = int(270.0 / PC_UNIT_SECTORANGLE);
      int blind_cols = blind_end_idx - blind_start_idx;
      total_cells = desc.rows() * (desc.cols() - blind_cols);
    }

    if (ratio < 0.015f) {
      std::cout << "[Warning] " << type
                << " ScanContext info too low: " << std::fixed
                << std::setprecision(2) << ratio * 100 << "% (" << non_zero
                << "/" << total_cells << " cells). Skipping." << std::endl;
      return true;
    }
    return false;
  }

  // 用户 API：构建数据库
  // 返回 true 表示成功插入，false 表示因稀疏等原因跳过
  bool makeAndSaveScancontextAndKeys(pcl::PointCloud<SCPointType> &_scan_down) {
    Eigen::MatrixXd sc = makeScancontext(_scan_down);
    if (isSCSparse(sc, "KeyFrame")) {
      return false;
    }

    Eigen::MatrixXd ringkey = makeRingkeyFromScancontext(sc);
    Eigen::MatrixXd sectorkey = makeSectorkeyFromScancontext(sc);
    std::vector<float> polarcontext_invkey_vec = eig2stdvec(ringkey);

    polarcontexts_.push_back(sc);
    polarcontext_invkeys_.push_back(ringkey);
    polarcontext_vkeys_.push_back(sectorkey);
    polarcontext_invkeys_mat_.push_back(polarcontext_invkey_vec);
    return true;
  }

  // Eigen::MatrixXd makeRingkeyFromScancontext(Eigen::MatrixXd &_desc) {
  //   Eigen::MatrixXd invariant_key(_desc.rows(), 1);
  //   for (int row_idx = 0; row_idx < _desc.rows(); row_idx++) {
  //     Eigen::MatrixXd curr_row = _desc.row(row_idx);
  //     invariant_key(row_idx, 0) = curr_row.mean();
  //   }
  //   return invariant_key;
  // }

  Eigen::MatrixXd makeRingkeyFromScancontext(Eigen::MatrixXd &_desc) {
    Eigen::MatrixXd invariant_key(_desc.rows(), 1);

    int blind_start_idx = int(90.0 / PC_UNIT_SECTORANGLE);
    int blind_end_idx = int(270.0 / PC_UNIT_SECTORANGLE);

    for (int row_idx = 0; row_idx < _desc.rows(); row_idx++) {
      double sum = 0.0;
      int count = 0;

      for (int col_idx = 0; col_idx < _desc.cols(); col_idx++) {
        if (use_fov_mask_) {
          if (col_idx >= blind_start_idx && col_idx < blind_end_idx) {
            continue; 
          }
        }

        sum += _desc(row_idx, col_idx);
        count++;
      }

      if (count > 0) {
        invariant_key(row_idx, 0) = sum / (double)count;
      } else {
        invariant_key(row_idx, 0) = 0.0;
      }
    }

    return invariant_key;
  }

  Eigen::MatrixXd makeSectorkeyFromScancontext(Eigen::MatrixXd &_desc) {
    Eigen::MatrixXd variant_key(1, _desc.cols());
    for (int col_idx = 0; col_idx < _desc.cols(); col_idx++) {
      Eigen::MatrixXd curr_col = _desc.col(col_idx);
      variant_key(0, col_idx) = curr_col.mean();
    }
    return variant_key;
  }

  // 核心匹配算法
  double distDirectSC(MatrixXd &_sc1, MatrixXd &_sc2) {
    int num_eff_cols = 0;
    double sum_sector_similarity = 0;

    for (int col_idx = 0; col_idx < _sc1.cols(); col_idx++) {
      VectorXd col_sc1 = _sc1.col(col_idx);
      VectorXd col_sc2 = _sc2.col(col_idx);

      if (col_sc1.norm() == 0 || col_sc2.norm() == 0)
        continue;

      double sector_similarity =
          col_sc1.dot(col_sc2) / (col_sc1.norm() * col_sc2.norm());
      sum_sector_similarity += sector_similarity;
      num_eff_cols++;
    }

    if (num_eff_cols == 0)
      return 1.0;

    double sc_sim = sum_sector_similarity / num_eff_cols;
    return 1.0 - sc_sim;
  }

  int fastAlignUsingVkey(MatrixXd &_vkey1, MatrixXd &_vkey2) {
    int argmin_vkey_shift = 0;
    double min_veky_diff_norm = 10000000;
    for (int shift_idx = 0; shift_idx < _vkey1.cols(); shift_idx++) {
      MatrixXd vkey2_shifted = circshift(_vkey2, shift_idx);
      MatrixXd vkey_diff = _vkey1 - vkey2_shifted;
      double cur_diff_norm = vkey_diff.norm();
      if (cur_diff_norm < min_veky_diff_norm) {
        argmin_vkey_shift = shift_idx;
        min_veky_diff_norm = cur_diff_norm;
      }
    }
    return argmin_vkey_shift;
  }

  std::pair<double, int> distanceBtnScanContext(MatrixXd &_sc1,
                                                MatrixXd &_sc2) {
    // 1. 快速粗对齐
    MatrixXd vkey_sc1 = makeSectorkeyFromScancontext(_sc1);
    MatrixXd vkey_sc2 = makeSectorkeyFromScancontext(_sc2);
    int argmin_vkey_shift = fastAlignUsingVkey(vkey_sc1, vkey_sc2);

    const int SEARCH_RADIUS = round(0.5 * SEARCH_RATIO * _sc1.cols());
    std::vector<int> shift_idx_search_space{argmin_vkey_shift};
    for (int ii = 1; ii < SEARCH_RADIUS + 1; ii++) {
      shift_idx_search_space.push_back((argmin_vkey_shift + ii + _sc1.cols()) %
                                       _sc1.cols());
      shift_idx_search_space.push_back((argmin_vkey_shift - ii + _sc1.cols()) %
                                       _sc1.cols());
    }
    std::sort(shift_idx_search_space.begin(), shift_idx_search_space.end());

    // 2. 精细对齐
    int argmin_shift = 0;
    double min_sc_dist = 10000000;
    for (int num_shift : shift_idx_search_space) {
      MatrixXd sc2_shifted = circshift(_sc2, num_shift);
      // 这里的 distDirectSC 已经包含了盲区忽略逻辑
      double cur_sc_dist = distDirectSC(_sc1, sc2_shifted);
      if (cur_sc_dist < min_sc_dist) {
        argmin_shift = num_shift;
        min_sc_dist = cur_sc_dist;
      }
    }
    return std::make_pair(min_sc_dist, argmin_shift);
  }

  // [修改] 极坐标可视化函数：将 ScanContext 转换为圆形的极坐标展示，更直观
  cv::Mat getScanContextVisual(const Eigen::MatrixXd &_sc) {
    int rings = _sc.rows();
    int sectors = _sc.cols();
    int img_size = 400; // 可视化窗口大小
    cv::Mat gray_img = cv::Mat::zeros(img_size, img_size, CV_8UC1);
    cv::Point center(img_size / 2, img_size / 2);
    double max_pixel_radius = img_size / 2.0 - 10;

    // 像素级映射，将线性矩阵映射到极坐标圆盘
    for (int y = 0; y < img_size; y++) {
      for (int x = 0; x < img_size; x++) {
        double dx = x - center.x;
        double dy = center.y - y; // 图像系转为笛卡尔系
        double dist = std::sqrt(dx * dx + dy * dy);

        if (dist > max_pixel_radius || dist < 2.0)
          continue;

        // 计算环索引
        int r_idx = (int)(dist / max_pixel_radius * rings);
        if (r_idx >= rings)
          r_idx = rings - 1;

        // 计算扇区索引 (对应 xy2theta 的角度映射)
        double angle = std::atan2(dy, dx) * 180.0 / M_PI;
        if (angle < 0)
          angle += 360.0;

        int s_idx = (int)(angle / 360.0 * sectors);
        if (s_idx >= sectors)
          s_idx = sectors - 1;

        float val = _sc(r_idx, s_idx);
        // 映射高度到 0-255
        gray_img.at<uchar>(y, x) = (uchar)std::min(
            255.0, std::max(0.0, 255.0 * val / LIDAR_CEILING_HEIGHT));
      }
    }

    cv::Mat color_img;
    cv::applyColorMap(gray_img, color_img, cv::COLORMAP_JET);

    // 将背景（灰度为0且原本未被赋值的地方）设为纯黑，避免 JET 渲染成深蓝色
    for (int y = 0; y < img_size; y++) {
      for (int x = 0; x < img_size; x++) {
        if (gray_img.at<uchar>(y, x) == 0) {
          color_img.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 0, 0);
        }
      }
    }

    // [要求] 绘制扇区和环的隔线，呈现“靶子”效果
    cv::Scalar grid_color(70, 70, 70); // 灰色格线
    // 1. 绘制径向扇区隔线
    for (int i = 0; i < sectors; ++i) {
      double angle_rad = (i * 360.0 / sectors) * M_PI / 180.0;
      cv::Point pt_end(
          (int)(center.x + max_pixel_radius * std::cos(angle_rad)),
          (int)(center.y - max_pixel_radius * std::sin(angle_rad)));
      cv::line(color_img, center, pt_end, grid_color, 1, cv::LINE_AA);
    }
    // 2. 绘制圆环隔线
    for (int i = 1; i <= rings; ++i) {
      int r = (int)(i * max_pixel_radius / rings);
      cv::circle(color_img, center, r, grid_color, 1, cv::LINE_AA);
    }

    return color_img;
  }

  // 用户 API：查询
  // 返回: <MatchID, YawDiff(rad)>
  // MatchID 为 -1 表示未找到
  std::pair<int, float>
  detectLoopClosureID(pcl::PointCloud<SCPointType>::Ptr query_cloud) {
    int loop_id = -1;

    // 生成 Query 的描述子
    Eigen::MatrixXd curr_desc = makeScancontext(*query_cloud);
    isSCSparse(curr_desc, "QueryFrame");
    Eigen::MatrixXd curr_ringkey = makeRingkeyFromScancontext(curr_desc);
    Eigen::MatrixXd curr_sectorkey = makeSectorkeyFromScancontext(curr_desc);
    std::vector<float> curr_ringkey_vec = eig2stdvec(curr_ringkey);

    // 1. KDTree 最近邻搜索候选
    if (polarcontext_invkeys_mat_.size() < NUM_EXCLUDE_RECENT) {
      return std::make_pair(-1, 0.0);
    }

    // 更新 Tree (支持增量构建，这里每 50 帧完整重建一次)
    tree_making_period_conter++;
    if (tree_making_period_conter > TREE_MAKING_PERIOD_) {
      tree_making_period_conter = 0;
      polarcontext_invkeys_to_search_ = polarcontext_invkeys_mat_;
      polarcontext_tree_.reset();
      polarcontext_tree_ = std::make_unique<InvKeyTree>(
          PC_NUM_RING, polarcontext_invkeys_to_search_, 10);
    }

    if (!polarcontext_tree_) {
      polarcontext_tree_ = std::make_unique<InvKeyTree>(
          PC_NUM_RING, polarcontext_invkeys_mat_, 10);
    }

    std::vector<size_t> candidate_indices(NUM_CANDIDATES_FROM_TREE);
    std::vector<float> out_distances_sq(NUM_CANDIDATES_FROM_TREE);
    nanoflann::KNNResultSet<float> knn_results(NUM_CANDIDATES_FROM_TREE);
    knn_results.init(&candidate_indices[0], &out_distances_sq[0]);
    polarcontext_tree_->index->findNeighbors(knn_results, &curr_ringkey_vec[0],
                                             nanoflann::SearchParams(10));

    // 2. 在候选集中进行精细匹配
    double min_dist = 1000000;
    int best_align_angle = 0;

    for (int i = 0; i < NUM_CANDIDATES_FROM_TREE; i++) {
      int candidate_idx = candidate_indices[i];
      MatrixXd candidate_sc = polarcontexts_[candidate_idx];
      auto res = distanceBtnScanContext(curr_desc, candidate_sc);

      if (res.first < min_dist) {
        min_dist = res.first;
        best_align_angle = res.second;
        loop_id = candidate_idx;
      }
    }

    // 3. 阈值判断
    float yaw_diff_rad = 0.0;
    if (min_dist > SC_DIST_THRES) {
      loop_id = -1;
    } else {
      // 计算偏航角差异 (rad)
      // sc_manager->PC_UNIT_SECTORANGLE 是每个 sector 的度数
      double yaw_diff_deg = best_align_angle * PC_UNIT_SECTORANGLE;
      yaw_diff_rad = yaw_diff_deg * M_PI / 180.0;
      // 归一化到 [-pi, pi]
      if (yaw_diff_rad > M_PI)
        yaw_diff_rad -= 2 * M_PI;
    }

    return std::make_pair(loop_id, (float)yaw_diff_rad);
  }
};