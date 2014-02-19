find_package(OpenCV)

set(bin_dir "/home/lgao/Workspace/cxx/DPM_Detection/libfdpm/build")
set(src_dir "/home/lgao/Workspace/cxx/DPM_Detection/libfdpm")

set(OPENDPM_INCLUDE_DIRS 
	${src_dir}/src/latentsvm
	${src_dir}/src/libffld
	${src_dir}/src/libffld/CvLSVMRead)

INCLUDE_DIRECTORIES(${OPENDPM_INCLUDE_DIRS})

SET(OPENDPM_LIB_DIRS ${LIBRARY_OUTPUT_PATH})
LINK_DIRECTORIES(${OPENDPM_LIB_DIRS})

set(OPENDPM_LIBS libffld latentsvm)
