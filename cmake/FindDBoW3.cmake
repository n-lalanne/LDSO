# ===================================================================================
#  DBoW3 CMake configuration file
#
#             ** File generated automatically, do not modify **
#
#  Usage from an external project:
#    In your CMakeLists.txt, add these lines:
#
#    FIND_PACKAGE(DBoW3 REQUIRED )
#    TARGET_LINK_LIBRARIES(MY_TARGET_NAME )
#
#    This file will define the following variables:
#      - DBoW3_LIBS          : The list of libraries to links against.
#      - DBoW3_LIB_DIR       : The directory where lib files are. Calling LINK_DIRECTORIES
#                                with this path is NOT needed.
#      - DBoW3_VERSION       : The  version of this PROJECT_NAME build. Example: "1.2.0"
#      - DBoW3_VERSION_MAJOR : Major version part of VERSION. Example: "1"
#      - DBoW3_VERSION_MINOR : Minor version part of VERSION. Example: "2"
#      - DBoW3_VERSION_PATCH : Patch version part of VERSION. Example: "0"
#
# ===================================================================================

find_path(DBoW3_INCLUDE_DIRS DBoW3/BowVector.h
  $ENV{DBoW3_ROOT}/include
  /usr/local/include
  /usr/include
  /opt/local/include
  /sw/local/include
  /sw/include
  NO_DEFAULT_PATH
  )

   find_library(DBoW3_WIN32
    NAMES "DBoW3001"
    PATHS
    $ENV{DBoW3_ROOT}/lib
    )

	find_library(DBoW3_LINUX
    NAMES "DBoW3"
    PATHS
    ~/Library/Frameworks
    /Library/Frameworks
    /usr/local/lib
    /usr/local/lib64
    /usr/lib
    /usr/lib64
    /opt/local/lib
    /sw/local/lib
    /sw/lib
    )

if(DBoW3_WIN32)
	SET(DBoW3_LIBRARIES ${DBoW3_WIN32})
else()
	SET(DBoW3_LIBRARIES ${DBoW3_LINUX})
endif()



set(DBoW3_FOUND "NO")
if(DBoW3_LIBRARIES AND DBoW3_INCLUDE_DIRS)
  set(DBoW3_FOUND "YES")
  message("Found DBoW3: ${DBoW3_LIBRARIES}, ${DBoW3_INCLUDE_DIRS}")
else()
	MESSAGE(FATAL_ERROR "Could not find DBoW3.")
endif(DBoW3_LIBRARIES AND DBoW3_INCLUDE_DIRS)

SET(DBoW3_VERSION        0.0.1)
SET(DBoW3_VERSION_MAJOR  0)
SET(DBoW3_VERSION_MINOR  0)
SET(DBoW3_VERSION_PATCH  1)
