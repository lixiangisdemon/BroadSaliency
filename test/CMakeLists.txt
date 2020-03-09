project(mtest)

include_directories(${CMAKE_SOURCE_DIR}/3rdparty/densecrf/include)
include_directories(${CMAKE_SOURCE_DIR}/3rdparty//eigen3)
include_directories(${CMAKE_SOURCE_DIR}/featureExtra/include)
include_directories(${CMAKE_SOURCE_DIR}/supercrf/include)

if(WIN32)
    set(DENSECRF_LIBS 
        ${3RDPARTY}/densecrf/build/src/libdensecrf.dll
        ${3RDPARTY}/densecrf/build/src/liboptimization.dll
    )
elseif(NOT APPLE) 
    set(DENSECRF_LIBS 
        ${3RDPARTY}/densecrf/build/src/libdensecrf.so
        ${3RDPARTY}/densecrf/build/src/liboptimization.so
    )
else(WIN32)
    set(DENSECRF_LIBS 
    ${3RDPARTY}/densecrf/build/src/libdensecrf.dylib
    ${3RDPARTY}/densecrf/build/src/liboptimization.dylib
)
endif(WIN32)

FIND_PACKAGE(OpenCV REQUIRED)
add_executable(mtest test.cpp)
TARGET_LINK_LIBRARIES( mtest ${OpenCV_LIBS} ${DENSECRF_LIBS} featureExtra supercrf )
