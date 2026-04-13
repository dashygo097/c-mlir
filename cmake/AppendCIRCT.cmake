# Find and configure CIRCT for the project

if(NOT DEFINED CIRCT_DIR)
  message(FATAL_ERROR "CIRCT_DIR must be set to the CIRCT installation directory")
endif()

# Find CIRCT package
message(STATUS "Searching for CIRCTConfig.cmake in: ${CIRCT_DIR}")
find_package(CIRCT REQUIRED CONFIG PATHS ${CIRCT_DIR} NO_DEFAULT_PATH)

if(NOT CIRCT_FOUND)
  message(FATAL_ERROR "CIRCT not found in ${CIRCT_DIR}")
endif()

message(STATUS "Found CIRCT ${CIRCT_PACKAGE_VERSION}")
message(STATUS "Using CIRCTConfig.cmake in: ${CIRCT_DIR}")
message(STATUS "CIRCT Include Dir: ${CIRCT_INCLUDE_DIRS}")
message(STATUS "CIRCT Binary Dir: ${CIRCT_BINARY_DIR}")

# Set CIRCT-related variables
set(CIRCT_CMAKE_DIR ${CIRCT_DIR} CACHE STRING "Location of CIRCT CMake modules")

# Include CIRCT CMake modules
list(APPEND CMAKE_MODULE_PATH "${CIRCT_CMAKE_DIR}")
include(AddCIRCT)

# Add CIRCT include directories
include_directories(${CIRCT_INCLUDE_DIRS})

# Add CIRCT library directories
if(DEFINED CIRCT_LIBRARY_DIR)
  link_directories(${CIRCT_LIBRARY_DIR})
endif()

print_info("✓ CIRCT configured successfully\n" "32")
