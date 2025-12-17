cmake_minimum_required(VERSION 3.8)
project(cartesian_planner)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter)
find_package(rclpy REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "srv/PlanScanPath.srv"
  "action/PlanSpline.action"
  DEPENDENCIES geometry_msgs
)


install(
  PROGRAMS cartesian_planner/spline_planner.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME spline_planner
)


install(
  DIRECTORY cartesian_planner/
  DESTINATION lib/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages/cartesian_planner
  FILES_MATCHING PATTERN "*.py"
)


ament_export_dependencies(rosidl_default_runtime)

ament_package()
