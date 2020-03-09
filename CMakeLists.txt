cmake_minimum_required(VERSION 3.8)
project(BroadSaliency)

# setup flags
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")	# -std=gnu++11
set(3RDPARTY ${CMAKE_SOURCE_DIR}/3rdparty)

#set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/3rdparty/opencv)
set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR})

add_subdirectory(featureExtra)
add_subdirectory(supercrf)
add_subdirectory(test)

install(TARGETS featureExtra supercrf DESTINATION libs)