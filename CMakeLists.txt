cmake_minimum_required(VERSION 3.17)
project(solid_sim_matcher)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
#     # 强制 16 字节对齐以兼容 PCL
#     add_compile_options(-O2 -march=native)
#     add_definitions(-DEIGEN_MAX_ALIGN_BYTES=16) 
#     message(STATUS "Optimization flags: -O2 -march=native (Fixed Alignment)")
# endif()

# Dependencies
# 指定具体组件可以显著加快 find_package(PCL) 的搜索速度
find_package(PCL REQUIRED COMPONENTS common io filters surface visualization)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(octomap REQUIRED)
find_package(OpenMP)

# 现代 CMake 推荐使用 target-based 配置，减少全局变量开销
add_executable(match src/match.cpp)
add_executable(grid_feature_extractor src/grid_feature_extractor.cpp)

if(OpenMP_CXX_FOUND)
    target_link_libraries(match OpenMP::OpenMP_CXX)
    target_link_libraries(grid_feature_extractor OpenMP::OpenMP_CXX)
endif()

# 为 match 目标配置
target_include_directories(match PRIVATE 
    include
    ${PCL_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIRS}
    ${OCTOMAP_INCLUDE_DIRS}
)
target_link_libraries(match
    ${PCL_LIBRARIES}
    ${OpenCV_LIBS}
    octomap
    octomath
)
target_compile_definitions(match PRIVATE ${PCL_DEFINITIONS})

# 为 grid_feature_extractor 目标配置
target_include_directories(grid_feature_extractor PRIVATE 
    include
    ${PCL_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIRS}
    ${OCTOMAP_INCLUDE_DIRS}
)
target_link_libraries(grid_feature_extractor
    ${PCL_LIBRARIES}
    ${OpenCV_LIBS}
    octomap
    octomath
)
target_compile_definitions(grid_feature_extractor PRIVATE ${PCL_DEFINITIONS})