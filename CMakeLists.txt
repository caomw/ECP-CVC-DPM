# Open The CMake GUI
# specify the source directory and the binary directory
# press configure
project(DPM_OpenCV)

#-------------------------------------------------------------------------------
#version
set(OpenDPM_VERSION_MAJOR 0)
set(OpenDPM_VERSION_MINOR 1)
set(OpenDPM_VERSION_PATCH 0)
set(OpenDPM_VERSION ${OpenDPM_VERSION_MAJOR}.${OpenDPM_VERSION_MINOR}.${OpenDPM_VERSION_PATCH})

cmake_minimum_required(VERSION 2.6)

OPTION(FFLD_HOGPYRAMID_FELZENSZWALB_FEATURES "Use Felzenszwalb's original features (slower and not as accurate, provided for compatibility only)." OFF)
OPTION(FFLD_HOGPYRAMID_DOUBLE "Use doubles instead of floats (slower, uses twice more memory, and the increase in precision is not necessarily useful)." OFF)

IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build, options are: None (CMAKE_CXX_FLAGS or CMAKE_C_FLAGS used) Debug Release RelWithDebInfo MinSizeRel." FORCE)
ENDIF()

IF(FFLD_HOGPYRAMID_FELZENSZWALB_FEATURES)
  ADD_DEFINITIONS(-DFFLD_HOGPYRAMID_FELZENSZWALB_FEATURES)
ENDIF()

IF(FFLD_HOGPYRAMID_DOUBLE)
  ADD_DEFINITIONS(-DFFLD_HOGPYRAMID_DOUBLE)
ENDIF()

FILE(GLOB EIGEN_ARCHIVE "eigen*")
FIND_PATH(EIGEN_INCLUDE_DIR NAMES Eigen PATHS /usr/local/include/eigen3/)
IF(NOT EIGEN_INCLUDE_DIR)
   MESSAGE(FATAL_ERROR "Could not find Eigen.")
ENDIF()
INCLUDE_DIRECTORIES(${EIGEN_INCLUDE_DIR})

FIND_PATH(FFTW3_INCLUDE_DIR fftw3.h)
	FIND_LIBRARY(FFTW3_LIBRARIES NAMES fftw3f)
IF(NOT FFTW3_INCLUDE_DIR OR NOT FFTW3_LIBRARIES)
  MESSAGE(FATAL_ERROR "Could not find fftw3.")
ENDIF()
INCLUDE_DIRECTORIES(${FFTW3_INCLUDE_DIR})

FIND_PACKAGE(JPEG REQUIRED)
IF(JPEG_FOUND)
  INCLUDE_DIRECTORIES(${JPEG_INCLUDE_DIR})
  #TARGET_LINK_LIBRARIES(ffld ${JPEG_LIBRARIES})
ENDIF()

FIND_PACKAGE(LibXml2 REQUIRED)
IF(LIBXML2_FOUND)
  INCLUDE_DIRECTORIES(${LIBXML2_INCLUDE_DIR})
  #TARGET_LINK_LIBRARIES(ffld ${LIBXML2_LIBRARIES})
  ADD_DEFINITIONS(${LIBXML2_DEFINITIONS})
ENDIF()

FIND_PACKAGE(OpenCV REQUIRED)
IF(OPENCV_FOUND)
	#INCLUDE_DIRECTORIES(${OPENCV_INCLUDE_DIR})
	#TARGET_LINK_LIBRARIES(ffld ${OPENCV_LIBRARIES})
ENDIF()

# Not required, but stronlgy recommended on multi-core systems
FIND_PACKAGE(OpenMP)
IF(OPENMP_FOUND)
  SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS} -fPIC")
  SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS} -fPIC")
  SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_C_FLAGS} -lgomp")
ENDIF()

#option(BUILD_QOPENTLD "Build with Qt-config-dialog." OFF)
#option(USE_SYSTEM_LIBS "Use the installed version of libconfig++." OFF)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)

#-------------------------------------------------------------------------------
#add subdirectories

add_subdirectory(src/latentsvm)
add_subdirectory(src/libffld)
#add_subdirectory(src/libdpm_ecp)
#add_subdirectory(src/libutil)

configure_file("${PROJECT_SOURCE_DIR}/OpenDPMConfig.cmake.in" "${PROJECT_BINARY_DIR}/OpenDPMConfig.cmake" @ONLY)
