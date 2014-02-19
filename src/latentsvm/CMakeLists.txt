link_directories(
				../libffld
				../libffld/CvLSVMRead
				${OpenCV_LIB_DIR})

include_directories(
				../libffld
				../libffld/CvLSVMRead
				${OpenCV_INCLUDE_DIRS})

#---------------------------------------------------
add_library(	
				lsvm_exe
				CLSvmDetector.h
				CLSvmDetector.cpp
				latentsvm_multidetect.cpp )

target_link_libraries(	lsvm_exe
						libffld
						${OpenCV_LIBS})

#---------------------------------------------------
add_executable(dpm_detector latentsvm_multidetect.cpp)

target_link_libraries(	dpm_detector 
						lsvm_exe 
						libffld
						${OpenCV_LIBS})

install(TARGETS lsvm_exe DESTINATION bin)