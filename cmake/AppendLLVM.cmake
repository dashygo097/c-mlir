# Find and configure LLVM for the project

if(NOT DEFINED LLVM_DIR)
  message(FATAL_ERROR "LLVM_DIR must be set to the LLVM installation directory")
endif()

# Find LLVM package
find_package(LLVM REQUIRED CONFIG PATHS ${LLVM_DIR} NO_DEFAULT_PATH)

if(NOT LLVM_FOUND)
  message(FATAL_ERROR "LLVM not found in ${LLVM_DIR}")
endif()

message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
message(STATUS "LLVM Binary Dir: ${LLVM_BINARY_DIR}")
message(STATUS "LLVM Include Dir: ${LLVM_INCLUDE_DIRS}")
message(STATUS "LLVM Library Dir: ${LLVM_LIBRARY_DIR}")

# Set LLVM-related variables
set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
set(LLVM_CMAKE_DIR ${LLVM_DIR})

# Include LLVM CMake modules
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

# Include required LLVM modules
include(TableGen)
include(AddLLVM)
include(HandleLLVMOptions)

# Add LLVM include directories
include_directories(${LLVM_INCLUDE_DIRS})

# Add LLVM library directories
link_directories(${LLVM_LIBRARY_DIR})

# Add LLVM definitions
add_definitions(${LLVM_DEFINITIONS})

# Set C++ standard based on LLVM if not already set
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

# Export LLVM variables for use in the project
set(LLVM_SOURCE_DIR ${LLVM_BUILD_MAIN_SRC_DIR} CACHE STRING "Location of LLVM source")

print_info("✓ LLVM configured successfully\n" "32")
