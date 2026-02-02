#pragma once

#include <algorithm>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <vector>

class BEVDebugger {
public:
  // --- 核心显示参数 ---
  int WINDOW_SIZE = 1000;      // 窗口的像素大小 (比如 1000x1000)
  float DISPLAY_RANGE = 80.0f; // 你希望在窗口中看到的物理半径 (米)
                               // 如果你想看细节，就设小一点，比如 20.0
                               // 如果你想看 SC 全貌，就设为 SC 的最大半径

  // --- ScanContext 网格参数 (仅用于画线，不影响缩放) ---
  int SC_NUM_RINGS = 20;
  int SC_NUM_SECTORS = 60;

  /**
   * @brief 显示 BEV 差异图
   * @param query_cloud  红色 (Query)
   * @param gt_cloud     绿色 (Ground Truth)
   * @param display_radius 本次可视化的物理半径。如果不传，默认使用类的
   * DISPLAY_RANGE
   */
  void showDifference(const pcl::PointCloud<pcl::PointXYZ>::Ptr &query_cloud,
                      const pcl::PointCloud<pcl::PointXYZ>::Ptr &gt_cloud,
                      float display_radius = -1.0f,
                      const std::string &window_name = "BEV Debugger") {
    // 1. 确定可视化的物理范围
    float radius = (display_radius > 0) ? display_radius : DISPLAY_RANGE;

    // 2. 自动计算缩放比例 (像素/米)
    // 我们希望 radius 米的距离对应 WINDOW_SIZE / 2 个像素
    float scale = (WINDOW_SIZE / 2.0f) / radius;

    // 3. 创建画布 (黑色背景)
    cv::Mat bev_map = cv::Mat::zeros(WINDOW_SIZE, WINDOW_SIZE, CV_8UC3);

    // 4. 创建浮点型图层用于累加高度 (初始化为极小值)
    cv::Mat q_layer = cv::Mat::ones(WINDOW_SIZE, WINDOW_SIZE, CV_32F) * -999.0f;
    cv::Mat g_layer = cv::Mat::ones(WINDOW_SIZE, WINDOW_SIZE, CV_32F) * -999.0f;

    // 5. 投影点云
    projectToGrid(query_cloud, q_layer, scale);
    projectToGrid(gt_cloud, g_layer, scale);

    // 6. 渲染颜色
    float min_vis_z = -2.0f; // 可视化的高度下限
    float max_vis_z = 5.0f;  // 可视化的高度上限

    for (int r = 0; r < WINDOW_SIZE; ++r) {
      for (int c = 0; c < WINDOW_SIZE; ++c) {
        float q_z = q_layer.at<float>(r, c);
        float g_z = g_layer.at<float>(r, c);

        uchar red = 0, green = 0;

        // Query (Red)
        if (q_z > -900.0f) {
          float norm = std::clamp((q_z - min_vis_z) / (max_vis_z - min_vis_z),
                                  0.0f, 1.0f);
          red = static_cast<uchar>(80 + norm * 175); // 基础亮度80
        }
        // GT (Green)
        if (g_z > -900.0f) {
          float norm = std::clamp((g_z - min_vis_z) / (max_vis_z - min_vis_z),
                                  0.0f, 1.0f);
          green = static_cast<uchar>(80 + norm * 175);
        }

        // 合成 (Yellow if overlap)
        bev_map.at<cv::Vec3b>(r, c) = cv::Vec3b(0, green, red); // B, G, R
      }
    }

    // 7. 绘制 ScanContext 参考网格
    drawSCGrid(bev_map, scale, radius);

    // 8. 绘制中心十字和比例尺文字
    cv::line(bev_map, cv::Point(WINDOW_SIZE / 2 - 10, WINDOW_SIZE / 2),
             cv::Point(WINDOW_SIZE / 2 + 10, WINDOW_SIZE / 2),
             cv::Scalar(255, 255, 255), 1);
    cv::line(bev_map, cv::Point(WINDOW_SIZE / 2, WINDOW_SIZE / 2 - 10),
             cv::Point(WINDOW_SIZE / 2, WINDOW_SIZE / 2 + 10),
             cv::Scalar(255, 255, 255), 1);

    std::string info = "Range: " + std::to_string((int)radius) + "m";
    cv::putText(bev_map, info, cv::Point(20, WINDOW_SIZE - 20),
                cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 2);

    cv::putText(bev_map, "RED: Query | GREEN: GT", cv::Point(20, 30),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 255), 1);

    // 9. 显示
    cv::namedWindow(window_name, cv::WINDOW_NORMAL); // 允许调整窗口大小
    cv::imshow(window_name, bev_map);

    // 这里的 waitKey 放在外面调用更灵活，或者这里只用 waitKey(1)
    // 如果想暂停：
    std::cout
        << "[BEV Debugger] Press any key on the image window to continue..."
        << std::endl;
    cv::waitKey(0);
  }

private:
  void projectToGrid(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
                     cv::Mat &grid, float scale) {
    int center = WINDOW_SIZE / 2;

    for (const auto &pt : cloud->points) {
      // 坐标变换：
      // 图像 Row (Y) 对应 物理 -X (因为图像Y向下，雷达X向前)
      // 图像 Col (X) 对应 物理 -Y (因为图像X向右，雷达Y向左)
      // 这样 图像上方=雷达前方，图像左方=雷达左方

      int row = center - static_cast<int>(pt.x * scale);
      int col = center - static_cast<int>(pt.y * scale);

      if (row >= 0 && row < WINDOW_SIZE && col >= 0 && col < WINDOW_SIZE) {
        // 取最高点 (ScanContext 逻辑)
        if (pt.z > grid.at<float>(row, col)) {
          grid.at<float>(row, col) = pt.z;
        }
      }
    }
  }

  void drawSCGrid(cv::Mat &img, float scale, float max_r_vis) {
    cv::Point center(WINDOW_SIZE / 2, WINDOW_SIZE / 2);

    // 假设 SC 的最大半径是固定的（比如80米），我们需要按比例画圆
    // 注意：SC_NUM_RINGS 是针对 MAX_RANGE 的划分
    // 我们需要知道 SC 实际定义的 Max Range 是多少，这里假设和 display_radius
    // 一样，或者你需要单独传入
    // 为了调试方便，我们这里画出的网格是基于当前视野范围的均匀网格

    // 绘制同心圆
    for (int i = 1; i <= SC_NUM_RINGS; ++i) {
      float r_m = (max_r_vis / SC_NUM_RINGS) * i; // 每一环的物理半径
      int r_px = static_cast<int>(r_m * scale);
      if (r_px > WINDOW_SIZE)
        break;
      cv::circle(img, center, r_px, cv::Scalar(80, 80, 80), 1);
    }

    // 绘制扇区线
    for (int i = 0; i < SC_NUM_SECTORS; ++i) {
      float angle_deg = (360.0f / SC_NUM_SECTORS) * i;
      float rad = angle_deg * M_PI / 180.0f;

      // 为了视觉习惯，旋转 -90度，让0度在正上方 (对应 x轴)
      // 但 math 是从 x轴(右)开始逆时针。
      // 雷达系：0度是X轴(前)。
      // 图像系：我们将 前 映射到了 Up。
      // 所以直接用 sin/cos 计算终点偏移即可

      // 图像坐标系修正:
      // X_img = center + (-y_lidar) * scale
      // Y_img = center - (x_lidar) * scale
      // x_lidar = r * cos(theta), y_lidar = r * sin(theta)

      float x_lidar = max_r_vis * std::cos(rad);
      float y_lidar = max_r_vis * std::sin(rad);

      int u = center.x - static_cast<int>(y_lidar * scale);
      int v = center.y - static_cast<int>(x_lidar * scale);

      cv::line(img, center, cv::Point(u, v), cv::Scalar(80, 80, 80), 1);
    }
  }
};