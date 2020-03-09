# Install script for directory: /Users/lixiang/Desktop/BroadSaliency/supercrf

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/Users/lixiang/Desktop/BroadSaliency")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/Users/lixiang/Desktop/BroadSaliency/libs/libdensecrf.dylib;/Users/lixiang/Desktop/BroadSaliency/libs/liboptimization.dylib")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
file(INSTALL DESTINATION "/Users/lixiang/Desktop/BroadSaliency/libs" TYPE FILE FILES
    "/Users/lixiang/Desktop/BroadSaliency/3rdparty/densecrf/build/src/libdensecrf.dylib"
    "/Users/lixiang/Desktop/BroadSaliency/3rdparty/densecrf/build/src/liboptimization.dylib"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/Users/lixiang/Desktop/BroadSaliency/libs/libsupercrf.dylib")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
file(INSTALL DESTINATION "/Users/lixiang/Desktop/BroadSaliency/libs" TYPE SHARED_LIBRARY FILES "/Users/lixiang/Desktop/BroadSaliency/build/supercrf/libsupercrf.dylib")
  if(EXISTS "$ENV{DESTDIR}/Users/lixiang/Desktop/BroadSaliency/libs/libsupercrf.dylib" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}/Users/lixiang/Desktop/BroadSaliency/libs/libsupercrf.dylib")
    execute_process(COMMAND /usr/bin/install_name_tool
      -delete_rpath "/Users/lixiang/Desktop/BroadSaliency/build/featureExtra"
      "$ENV{DESTDIR}/Users/lixiang/Desktop/BroadSaliency/libs/libsupercrf.dylib")
    execute_process(COMMAND /usr/bin/install_name_tool
      -delete_rpath "/Users/lixiang/opt/anaconda3/lib"
      "$ENV{DESTDIR}/Users/lixiang/Desktop/BroadSaliency/libs/libsupercrf.dylib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" -x "$ENV{DESTDIR}/Users/lixiang/Desktop/BroadSaliency/libs/libsupercrf.dylib")
    endif()
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

