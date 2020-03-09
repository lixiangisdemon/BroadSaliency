project(featureExtra)

include_directories(./include)
file(GLOB Sources src/*.cpp src/*.c )
add_library(featureExtra SHARED ${Sources}) #SHARED

FIND_PACKAGE(OpenCV REQUIRED)
TARGET_LINK_LIBRARIES(featureExtra ${OpenCV_LIBS} pthread)
install(TARGETS featureExtra DESTINATION ${CMAKE_SOURCE_DIR}/libs)