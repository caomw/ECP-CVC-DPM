link_directories(${OpenCV_LIB_DIR})
include_directories(${OpenCV_INCLUDE_DIRS})

add_library( libffld
		     CFFLD.h
		     CFFLD.cpp
			 Rectangle.h 
			 JPEGImage.h
			 Object.h
			 Scene.h
			 HOGPyramid.h
			 Patchwork.h
			 Model.h 
			 Mixture.h
			 Intersector.h
			 Rectangle.cpp 
			 JPEGImage.cpp 
			 Object.cpp 
			 Scene.cpp 
			 HOGPyramid.cpp 
			 Patchwork.cpp 
			 Model.cpp 
			 Mixture.cpp 
			 #CvLSVMRead/_lsvm_error.h
			 #CvLSVMRead/_lsvm_types.h
			 CvLSVMRead/_lsvmparser.h
			 CvLSVMRead/lsvmparser.cpp
			 # CvLSVMRead/precomp.cpp
			 # CvLSVMRead/precomp.hpp
			)
target_link_libraries( libffld 
					   ${FFTW3_LIBRARIES} 
					   ${JPEG_LIBRARIES}
					   ${LIBXML2_LIBRARIES}
					   ${OpenCV_LIBS})

# ADD_EXECUTABLE(ffld test.cpp)

#target_link_libraries( ffld
#                       lib_ffld 
#                       ${OpenCV_LIBS} )
#install(TARGETS ffld DESTINATION bin)
