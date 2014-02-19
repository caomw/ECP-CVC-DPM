link_directories(${OpenCV_LIB_DIR})

include_directories(${OpenCV_INCLUDE_DIRS})

#---------------------------------------------------
add_library( libutil
		     CStopwatch.h
		     CStopwatch.cpp )

target_link_libraries(libutil ${OpenCV_LIBS})


