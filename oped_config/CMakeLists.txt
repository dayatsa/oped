cmake_minimum_required(VERSION 3.0.2)
project(oped_config)

find_package(catkin REQUIRED COMPONENTS)

catkin_package()

install(DIRECTORY config launch maps worlds
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)