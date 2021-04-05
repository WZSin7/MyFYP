cmake_minimum_required(VERSION 2.8.3)
project(vocus2_ros)


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=c++0x -fopenmp")
set(CMAKE_CXX_FLAGS "-O2")
## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  dynamic_reconfigure
  image_transport
  image_geometry
  roscpp
  rospy
  sensor_msgs
  std_msgs
  message_generation
)
find_package(OpenCV REQUIRED)
find_package(Boost COMPONENTS serialization REQUIRED)

generate_dynamic_reconfigure_options(
  cfg/vocus2_ros.cfg
  #...
)

## Generate messages in the 'msg' folder
add_message_files(
   FILES
   BoundingBoxes.msg
   BoundingBox.msg
   GazeInfoBino_Array.msg
   Result.msg
   Result_Detectron2.msg
   forDemo.msg
 )

## Generate added messages and services with any dependencies listed here
 generate_messages(
   DEPENDENCIES
   std_msgs
   sensor_msgs
 )
catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES vocus
#  CATKIN_DEPENDS cv_bridge dynamic_reconfigure image_transport roscpp rospy sensor_msgs std_msgs
#  DEPENDS system_lib
)

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
)

link_directories(
  ${OpenCV_LIBRARY_DIRS}
  ${Boost_LIBRARY_DIRS}
)

 add_executable(vocus2_ros_node src/vocus2_ros_node.cpp src/VOCUS2.cpp src/ImageFunctions.h src/HelperFunctions.h src/ImageFunctions.cpp src/HelperFunctions.cpp src/VOCUS2.h src/VOCUS_ROS.h src/VOCUS_ROS.cpp)

 target_link_libraries(vocus2_ros_node
   ${catkin_LIBRARIES}
   ${OpenCV_LIBRARIES}
   ${Boost_LIBRARIES}

 )

add_dependencies(vocus2_ros_node ${PROJECT_NAME}_gencfg)

install(TARGETS vocus2_ros_node DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

catkin_install_python(PROGRAMS scripts/pr2_look_around.py DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

install(DIRECTORY launch DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})
