# Find and configure Clang for the project

if(NOT DEFINED Clang_DIR)
  message(FATAL_ERROR "Clang_DIR must be set to the Clang installation directory")
endif()

# Find Clang package
message(STATUS "Searching for ClangConfig.cmake in: ${Clang_DIR}")
find_package(Clang REQUIRED CONFIG PATHS ${Clang_DIR} NO_DEFAULT_PATH)

if(NOT Clang_FOUND)
  message(FATAL_ERROR "Clang not found in ${Clang_DIR}")
endif()

message(STATUS "Found Clang ${CLANG_PACKAGE_VERSION}")
message(STATUS "Using ClangConfig.cmake in: ${Clang_DIR}")
message(STATUS "Clang Include Dir: ${CLANG_INCLUDE_DIRS}")
message(STATUS "Clang Binary Dir: ${CLANG_BINARY_DIR}")

# Set Clang-related variables
set(CLANG_CMAKE_DIR ${Clang_DIR} CACHE STRING "Location of Clang CMake modules")

# Include Clang CMake modules
list(APPEND CMAKE_MODULE_PATH "${CLANG_CMAKE_DIR}")
include(AddClang)

# Add Clang include directories
include_directories(${CLANG_INCLUDE_DIRS})

# Add Clang library directories
if(DEFINED CLANG_LIBRARY_DIR)
  link_directories(${CLANG_LIBRARY_DIR})
endif()

print_info("✓ Clang configured successfully\n" "32")
