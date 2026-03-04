cmake_minimum_required(VERSION 3.17)
project(place_recognition_preprocess)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(PCL REQUIRED COMPONENTS common io filters surface visualization)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(octomap REQUIRED)
find_package(OpenMP)

add_executable(match_test src/match_test.cpp)
add_executable(simulator_test src/simulator_test.cpp)
add_executable(keyframe_simulator src/keyframe_simulator.cpp)

macro(configure target_name)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(${target_name} OpenMP::OpenMP_CXX)
    endif()

    target_include_directories(${target_name} PRIVATE 
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${PCL_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
        ${EIGEN3_INCLUDE_DIRS}
        ${OCTOMAP_INCLUDE_DIRS}
    )

    target_link_libraries(${target_name}
        ${PCL_LIBRARIES}
        ${OpenCV_LIBS}
        octomap
        octomath
    )

    target_compile_definitions(${target_name} PRIVATE 
        ${PCL_DEFINITIONS}
        PROJECT_ROOT_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
    )
endmacro()

configure(match_test)
configure(keyframe_simulator)
configure(simulator_test)