link_directories(${OpenCV_LIB_DIR}
				../libutil )

include_directories(${OpenCV_INCLUDE_DIRS}
					../libutil )

#---------------------------------------------------
add_library(libdpm
		cascadedetect.cpp
		cascadedetect.hpp
		datamatrix.cpp
		distancetransform.cpp
		featurepyramid.cpp
		fft.cpp
		hog.cpp
		latentsvm.cpp
		latentsvmdetector.cpp
		_latentsvm.h
		linemod.cpp
		_lsvm_distancetransform.h
		_lsvm_error.h
		_lsvm_fft.h
		_lsvm_matching.h
		lsvmparser.cpp
		_lsvmparser.h
		_lsvm_resizeimg.h
		_lsvm_routine.h
		lsvmtbbversion.cpp
		_lsvm_tbbversion.h
		_lsvm_types.h
		matching.cpp
		normal_lut.i
		objdetect.hpp
		objdetect_init.cpp
		precomp.cpp
		precomp.hpp
		resizeimg.cpp
		routine.cpp)

target_link_libraries(libdpm libutil ${OpenCV_LIBS})


