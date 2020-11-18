
cmake_minimum_required(VERSION 3.1)

project(tp1)

set(CMAKE_BUILD_TYPE DEBUG)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(OpenCV REQUIRED)
# find_package(Threads REQUIRED)

set(CMAKE_CXX_STANDARD 11)

file(GLOB INCLUDES include/*.h)

file(GLOB SOURCES src/*.cpp)

add_library( BaseFiles ${SOURCES} ${INCLUDES} )


include_directories(
  ${OpenCV_INCLUDE_DIRS}
  include/
)

# add_executable(
#   ${CMAKE_PROJECT_NAME}
#   ${SOURCES}
# )

# target_link_libraries(
#   ${CMAKE_PROJECT_NAME}
#   ${OpenCV_LIBS}
#   ${CMAKE_THREAD_LIBS_INIT}
# )

set(TP1 "tp1")

add_executable(
  TP1
  mains/tp1.cpp
)

target_link_libraries(
  TP1
  BaseFiles
  ${OpenCV_LIBS}
)

set(TP2 "tp2")
add_executable(
  TP2
  mains/tp2.cpp
)

target_link_libraries(
  TP2
  BaseFiles
  ${OpenCV_LIBS}
)


configure_file(
  datas/square_sample_0.png ${CMAKE_CURRENT_BINARY_DIR}/datas/square_sample_0.png COPYONLY
)
configure_file(
  datas/square_sample_1.png ${CMAKE_CURRENT_BINARY_DIR}/datas/square_sample_1.png COPYONLY
)
configure_file(
  datas/square_sample_2.png ${CMAKE_CURRENT_BINARY_DIR}/datas/square_sample_2.png COPYONLY
)
configure_file(
  datas/square_sample.png ${CMAKE_CURRENT_BINARY_DIR}/datas/square_sample.png COPYONLY
)
configure_file(
  datas/mr_piuel.jpeg ${CMAKE_CURRENT_BINARY_DIR}/datas/mr_piuel.jpeg COPYONLY
)
configure_file(
  datas/Palpa1.jpg ${CMAKE_CURRENT_BINARY_DIR}/datas/Palpa1.jpg COPYONLY
)
configure_file(
  datas/Palpa2.jpg ${CMAKE_CURRENT_BINARY_DIR}/datas/Palpa2.jpg COPYONLY
)
configure_file(
  datas/Lenna.png ${CMAKE_CURRENT_BINARY_DIR}/datas/Lenna.png COPYONLY
)
