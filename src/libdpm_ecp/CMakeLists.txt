link_directories(${OpenCV_LIB_DIR}
				../libutil )

include_directories(${OpenCV_INCLUDE_DIRS}
					../libutil )

#---------------------------------------------------
add_library(libdpm_ecp
		CLSVMDetector.h
		CLSVMDetector.cpp
		LibPartFilter.h
		LibPartFilter.cpp
		LibPyramid.h
		LibPyramid.cpp
		LibLSVM.hpp
		LibLSVM.cpp)

target_link_libraries(libdpm_ecp libutil ${OpenCV_LIBS})


