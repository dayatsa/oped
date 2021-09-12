cmake_minimum_required(VERSION 3.0.2)
project(oped_teleopp)

find_package(catkin REQUIRED)

catkin_package()

install(DIRECTORY launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)


catkin_install_python(PROGRAMS 
  scripts/floor_controller.py 
  scripts/quadruped_controller.py 
  scripts/set_action.py 
  scripts/train.py 
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)