cmake_minimum_required(VERSION 3.17)
project(solid_sim_matcher)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Dependencies
find_package(PCL REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(octomap REQUIRED)

include_directories(
    include
    ${PCL_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIRS}
    ${OCTOMAP_INCLUDE_DIRS}
)

link_directories(
    ${PCL_LIBRARY_DIRS}
    ${OCTOMAP_LIBRARY_DIRS}
)

add_definitions(${PCL_DEFINITIONS})

add_executable(match src/match.cpp)
add_executable(grid_feature_extractor src/grid_feature_extractor.cpp)

target_link_libraries(match
    ${PCL_LIBRARIES}
    ${OpenCV_LIBS}
    octomap
    octomath
)

target_link_libraries(grid_feature_extractor
    ${PCL_LIBRARIES}
    ${OpenCV_LIBS}
    octomap
    octomath
)
