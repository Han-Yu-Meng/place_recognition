// keyframe_simulator.cpp

#include "lidar_simulator.hpp"
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <queue>
#include <thread>

const std::string DATASET_DIR = "BaoLi";

namespace fs = std::filesystem;

struct ExtractTask {
  double world_x, world_y;
};

LidarSimulator *g_simulator;
std::string g_output_dir;
int g_id_counter = 100000;
double g_robot_height = 0.325;

std::queue<ExtractTask> g_task_queue;
std::mutex g_queue_mutex;
std::condition_variable g_queue_cv;
std::atomic<bool> g_running{true};
std::atomic<int> g_total_tasks{0};
std::atomic<int> g_finished_tasks{0};

// 消费者线程：处理计算任务
void processTasks() {
  while (g_running) {
    ExtractTask task;
    {
      std::unique_lock<std::mutex> lock(g_queue_mutex);
      g_queue_cv.wait(lock, [] { return !g_task_queue.empty() || !g_running; });

      if (!g_running && g_task_queue.empty())
        return;

      task = g_task_queue.front();
      g_task_queue.pop();
    }

    // --- 执行耗时的 Ray Casting ---
    if (!g_simulator)
      continue;

    double ground_z;
    bool ground_found = g_simulator->find_ground_height(
        task.world_x, task.world_y, ground_z, 2.0);

    bool success = false;
    if (ground_found &&
        (ground_z + g_robot_height <= 2.5)) { // 稍微放宽一点高度check
      bool any_saved = false;
      for (int yaw_deg = 0; yaw_deg < 360; yaw_deg += 45) {
        LidarSimulator::Pose pose;
        pose.x = task.world_x;
        pose.y = task.world_y;
        pose.z = ground_z + g_robot_height;
        pose.roll = 0;
        pose.pitch = 0;
        pose.yaw = yaw_deg * M_PI / 180.0;

        auto full_scan = g_simulator->simulate_scan(pose);

        if (full_scan->points.size() < 500)
          continue; // 松一点

        std::stringstream ss;
        ss << std::setw(6) << std::setfill('0') << g_id_counter;
        std::string base_name = ss.str();
        std::string pcd_filename = g_output_dir + base_name + ".pcd";
        std::string odom_filename = g_output_dir + base_name + ".odom";

        pcl::io::savePCDFileBinary(pcd_filename, *full_scan);

        // 保存 Odom
        std::ofstream odom_file(odom_filename);
        odom_file << std::fixed << std::setprecision(6);
        Eigen::Matrix3d R =
            Eigen::AngleAxisd(pose.yaw, Eigen::Vector3d::UnitZ())
                .toRotationMatrix();
        odom_file << R(0, 0) << " " << R(0, 1) << " " << R(0, 2) << " "
                  << pose.x << "\n";
        odom_file << R(1, 0) << " " << R(1, 1) << " " << R(1, 2) << " "
                  << pose.y << "\n";
        odom_file << R(2, 0) << " " << R(2, 1) << " " << R(2, 2) << " "
                  << pose.z << "\n";
        odom_file << 0.0 << " " << 0.0 << " " << 0.0 << " " << 1.0 << "\n";
        odom_file.close();
        
        g_id_counter++;
        any_saved = true;
      }
      success = any_saved;
    }

    g_finished_tasks++;
    
    // 进度条渲染
    {
      static std::mutex console_mutex;
      std::lock_guard<std::mutex> lock(console_mutex);
      const int bar_width = 50;
      float progress = (float)g_finished_tasks / g_total_tasks;
      int pos = bar_width * progress;

      std::cout << "\r[";
      for (int i = 0; i < bar_width; ++i) {
        if (i < pos) std::cout << "=";
        else if (i == pos) std::cout << ">";
        else std::cout << " ";
      }
      std::cout << "] " << int(progress * 100.0) << "% (" 
                << g_finished_tasks << "/" << g_total_tasks << ")" << std::flush;
    }
  }
}

int main(int argc, char **argv) {
  fs::path root(PROJECT_ROOT_DIR);
  
  // 从 scripts/DATASET 读取数据集名字
  std::string dataset_name = "BaoLi";
  std::ifstream dataset_file(root / "scripts" / "DATASET");
  if (dataset_file.is_open()) {
    dataset_file >> dataset_name;
    dataset_file.close();
  }
  
  fs::path dataset_dir = root / dataset_name;
  std::string map_path = (dataset_dir / "global_map.pcd").string();
  fs::path output_path = dataset_dir / "keyframes" / "simulated";
  g_output_dir = output_path.string() + "/";

  // 每次运行前删除并重新创建 simulated 文件夹
  if (fs::exists(output_path)) {
    fs::remove_all(output_path);
    std::cout << "[Grid] Cleaned folder: " << output_path << std::endl;
  }
  fs::create_directories(output_path);

  LidarSimulator simulator;
  simulator.load_config((dataset_dir / "lidar_config.yaml").string());
  if (!simulator.load_map(map_path, 0.10)) {
    return -1;
  }
  g_simulator = &simulator;

  // 读取 tasks.json 放入队列
  fs::path tasks_path = dataset_dir / "tasks.json";
  if (!fs::exists(tasks_path)) {
    std::cerr << "[Error] tasks.json not found: " << tasks_path << std::endl;
    return -1;
  }

  std::ifstream f(tasks_path);
  std::string line;
  while (std::getline(f, line)) {
    if (line.find("\"world_x\"") != std::string::npos) {
      double wx = std::stod(line.substr(line.find(":") + 1));
      std::getline(f, line);
      double wy = std::stod(line.substr(line.find(":") + 1));
      ExtractTask task;
      task.world_x = wx;
      task.world_y = wy;
      {
        std::lock_guard<std::mutex> lock(g_queue_mutex);
        g_task_queue.push(task);
        g_total_tasks++;
      }
    }
  }

  std::cout << "[Grid] Loaded " << g_total_tasks << " tasks from tasks.json" << std::endl;

  std::vector<std::thread> workers;
  // 限制线程数为 1，以解决 LidarSimulator 内部状态（T_body_lidar_）非线程安全导致的 Bus Error
  int num_threads = 1; 
  std::cout << "[Grid] Using " << num_threads << " thread to avoid race conditions." << std::endl;
  for (int i = 0; i < num_threads; ++i) {
    workers.emplace_back(processTasks);
  }

  // 等待任务完成
  while (g_finished_tasks < g_total_tasks) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }

  g_running = false;
  g_queue_cv.notify_all();
  for (auto &w : workers) {
    if (w.joinable()) w.join();
  }

  std::cout << "\n[Grid] All simulations finished. Saved to " << g_output_dir << std::endl;
  return 0;
}

