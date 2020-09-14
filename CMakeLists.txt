
cmake_minimum_required(VERSION 3.1)

project(tp1)

set(CMAKE_BUILD_TYPE DEBUG)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(OpenCV REQUIRED)
# find_package(Threads REQUIRED)

set(CMAKE_CXX_STANDARD 11)

file(GLOB INCLUDES include/*.h)

file(GLOB SOURCES src/*.cpp)


include_directories(
  ${OpenCV_INCLUDE_DIRS}
  include/  
)

add_executable(
  ${CMAKE_PROJECT_NAME}
  ${SOURCES}
)

target_link_libraries(
  ${CMAKE_PROJECT_NAME}
  ${OpenCV_LIBS}
#   ${CMAKE_THREAD_LIBS_INIT}
)

configure_file(
  datas/square_sample.png ${CMAKE_CURRENT_BINARY_DIR}/datas/square_sample.png COPYONLY
)
configure_file(
  datas/mr_piuel.jpeg ${CMAKE_CURRENT_BINARY_DIR}/datas/mr_piuel.jpeg COPYONLY
)