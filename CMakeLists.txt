cmake_minimum_required(VERSION 3.8)
project(cartesian_planner)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclpy REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "srv/PlanCartesianPath.srv"
  DEPENDENCIES geometry_msgs
)

install(
  PROGRAMS cartesian_planner/slerp_planner.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME slerp_planner
)




ament_export_dependencies(rosidl_default_runtime)

ament_package()
