cmake_minimum_required(VERSION 3.5)
project(quad_description)

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

# find dependencies
find_package(ament_cmake REQUIRED)

# Install Python executables
install(PROGRAMS
  # scripts/PID_ken.py
  # scripts/PID.py
  scripts/LQR.py
  # scripts/LQR_v1.py
  scripts/LQI.py
  scripts/iLQR.py
  scripts/MPCNode5.py
  # scripts/send_trajectory.py
  # scripts/send_trajectory_2.py
  scripts/send_traject.py
  scripts/trajectory_visualizer.py
  scripts/dashboard_realtime.py
  scripts/dashboard_analysis.py
  DESTINATION lib/${PROJECT_NAME}
)

################ INSTALL LAUNCH, ETC #################
install(DIRECTORY
  # add directories here
  urdf
  meshes
  launch 
  config
  worlds
  rviz
  DESTINATION share/${PROJECT_NAME})

ament_package()

