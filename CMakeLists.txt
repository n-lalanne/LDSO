cmake_minimum_required(VERSION 2.8)
project(ldso)

# Set default build type if not specified otherwise.
# See https://cmake.org/pipermail/cmake/2012-May/050243.html
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  message(STATUS "Setting build type to '${CMAKE_BUILD_TYPE}' as none was specified.")
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release"
    "MinSizeRel" "RelWithDebInfo")
endif()

# C++ standard
if(${CMAKE_VERSION} VERSION_LESS "3.2.0")
  set( CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS}" )
else()
  set(CMAKE_CXX_STANDARD 11)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_CXX_EXTENSIONS OFF)
endif()

# costum modules
LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

IF(MSVC)
	include(cotire)
ENDIF()

# TODO: change this to not output outside build dir
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)

# required libs
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Glog REQUIRED)
#find_package(Pangolin REQUIRED)
find_package(G2O REQUIRED)
find_package(DBoW3 REQUIRED)
find_package(Boost REQUIRED COMPONENTS system)

# optional libs
find_package(LibZip QUIET)

if (MSVC)
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc /FR /bigobj /fp:fast /openmp /Ox")
	add_definitions("-D__SSE3__")
	add_definitions("-DNOMINMAX")
	add_definitions("-D_USE_MATH_DEFINES")
	add_definitions("-DGOOGLE_GLOG_DLL_DECL=")
	add_definitions("-DGLOG_NO_ABBREVIATED_SEVERITIES=")
	add_definitions("-DWIN32_LEAN_AND_MEAN=")
else()
	add_definitions("-DENABLE_SSE")
	set(CMAKE_CXX_FLAGS "-Wall -Wno-deprecated -march=native -Wno-duplicate-decl-specifier -Wno-ignored-qualifiers -Wno-reorder -pthread")
endif (MSVC)

# decide if we have ziplib.
if (LIBZIP_LIBRARY)
  message("--- found ziplib (${LIBZIP_VERSION}), compiling with zip capability.")
  add_definitions(-DHAS_ZIPLIB=1)
  include_directories( ${LIBZIP_INCLUDE_DIR_ZIP} ${LIBZIP_INCLUDE_DIR_ZIPCONF} )
else()
  message("--- not found ziplib (${LIBZIP_LIBRARY}), compiling without zip capability.")
  set(LIBZIP_LIBRARY "")
endif()

include_directories(
        ${EIGEN3_INCLUDE_DIR}
        ${OpenCV_INCLUDE_DIR}
        ${GLOG_INCLUDE_DIR}
       # ${Pangolin_INCLUDE_DIRS}
		${G2O_INCLUDE_DIR}
		${DBoW3_INCLUDE_DIRS}
		${Boost_INCLUDE_DIR}

        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/thirdparty/Sophus
)

set( THIRD_PARTY_LIBS
        ${OpenCV_LIBS}
        ${GLOG_LIBRARY}
      #  ${Pangolin_LIBRARIES}

		${G2O_CORE_LIBRARY}
		${G2O_STUFF_LIBRARY}
		${G2O_TYPES_ICP}
		${G2O_TYPES_DATA}
		${G2O_TYPES_SBA}
		${G2O_TYPES_SCLAM2D}
		${G2O_TYPES_SIM3}
		${G2O_TYPES_SLAM3D}
		
		${G2O_SOLVER_EIGEN}

        ${DBoW3_LIBRARIES}
        ${LIBZIP_LIBRARY}
		${Boost_LIBRARY_DIRS}
)

if (MSVC)
	set( THIRD_PARTY_LIBS
        ${THIRD_PARTY_LIBS}
		-lpthread
	)
endif (MSVC)

add_subdirectory(src)
#add_subdirectory(examples)
