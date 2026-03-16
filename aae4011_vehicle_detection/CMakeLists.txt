cmake_minimum_required(VERSION 3.0.2)
project(aae4011_vehicle_detection)

find_package(catkin REQUIRED COMPONENTS
  rospy
  sensor_msgs
  std_msgs
  cv_bridge
  image_transport
  rosbag
)

catkin_package()

catkin_package()

install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

catkin_install_python(
  PROGRAMS
    scripts/extract_images.py
    scripts/vehicle_detector_node.py
    scripts/vehicle_detector_gui_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)