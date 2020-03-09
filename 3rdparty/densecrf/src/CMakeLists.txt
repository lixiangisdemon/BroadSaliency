include_directories( ../external/liblbfgs/include ../include )

add_library( densecrf SHARED util.cpp permutohedral.cpp unary.cpp pairwise.cpp objective.cpp densecrf.cpp labelcompatibility.cpp )
add_library( optimization SHARED optimization.cpp )
target_link_libraries( optimization lbfgs )
