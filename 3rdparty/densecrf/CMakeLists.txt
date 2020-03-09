cmake_minimum_required(VERSION 3.2)
project(pydensecrf)

# setup tensorRT flags
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")	# -std=gnu++11
include_directories(${CMAKE_SOURCE_DIR}/../eigen3)

add_subdirectory(external)
add_subdirectory(src)
