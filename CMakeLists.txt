cmake_minimum_required(VERSION 3.17)
project(solid_sim_matcher)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Dependencies
find_package(PCL REQUIRED COMPONENTS common io filters surface visualization)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(octomap REQUIRED)
find_package(OpenMP)

add_executable(match src/match.cpp)
add_executable(feature_selector src/feature_selector.cpp)

if(OpenMP_CXX_FOUND)
    target_link_libraries(match OpenMP::OpenMP_CXX)
    target_link_libraries(feature_selector OpenMP::OpenMP_CXX)
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
target_include_directories(feature_selector PRIVATE 
    include
    ${PCL_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIRS}
    ${OCTOMAP_INCLUDE_DIRS}
)
target_link_libraries(feature_selector
    ${PCL_LIBRARIES}
    ${OpenCV_LIBS}
    octomap
    octomath
)
target_compile_definitions(feature_selector PRIVATE ${PCL_DEFINITIONS})