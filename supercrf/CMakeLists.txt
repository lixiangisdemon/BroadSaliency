project(supercrf)

include_directories(./include)
include_directories(${CMAKE_SOURCE_DIR}/featureExtra/include)
include_directories(${3RDPARTY}/densecrf/include)
include_directories(${CMAKE_SOURCE_DIR}/3rdparty/eigen3)
file(GLOB Sources src/*.cpp src/*.c )
add_library(supercrf SHARED ${Sources})

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
TARGET_LINK_LIBRARIES(supercrf featureExtra ${OpenCV_LIBS} pthread ${DENSECRF_LIBS})
install(FILES ${DENSECRF_LIBS} DESTINATION ${CMAKE_SOURCE_DIR}/libs)
install(TARGETS supercrf DESTINATION ${CMAKE_SOURCE_DIR}/libs)