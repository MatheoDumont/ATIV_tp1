
cmake_minimum_required(VERSION 3.1)

project(tp1)

find_package(OpenCV REQUIRED)
# find_package(Threads REQUIRED)

set(CMAKE_CXX_STANDARD 11)

include_directories(
  ${OpenCV_INCLUDE_DIRS}
  include/
)

add_executable(
  ${CMAKE_PROJECT_NAME}
  src/main.cpp
#   src/analyse_image.cpp
)

target_link_libraries(
  ${CMAKE_PROJECT_NAME}
  ${OpenCV_LIBS}
#   ${CMAKE_THREAD_LIBS_INIT}
)
