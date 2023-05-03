cmake_minimum_required(VERSION 3.5)
project(tree_detect)

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
#find_package(tf_transformations REQUIRED)


## find_package(catkin REQUIRED COMPONENTS
##   geometry_msgs
##   roscpp
##   rospy
##   sensor_msgs
##   std_msgs
##   tf
## )

install(
  DIRECTORY
    launch
    config
    rviz
    scripts
  DESTINATION
    share/${PROJECT_NAME}/
)

install(
  PROGRAMS
  scripts/FRONTEND/detect_node.py
  scripts/FRONTEND/detect_node_openvino.py
  scripts/FRONTEND/detector.py
  scripts/FRONTEND/detector_openvino.py
  scripts/FRONTEND/measurement3d_2d.py
  scripts/BACKEND/slam_node.py
  scripts/BACKEND/EKF.py
  scripts/BACKEND/RangeBearing.py
  scripts/BACKEND/robot.py
  scripts/UTILS/visualize.py
  scripts/UTILS/visualize_slam.py
  scripts/MODELS/R-50_RGB_60k.pth
  scripts/MODELS/ResNext-101_fold_01.pth
  scripts/MODELS/model.bin
  scripts/MODELS/model.xml
  scripts/MODELS/model.onnx
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
