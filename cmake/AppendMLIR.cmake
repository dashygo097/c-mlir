# Find and configure MLIR for the project

if(NOT DEFINED MLIR_DIR)
  message(FATAL_ERROR "MLIR_DIR must be set to the MLIR installation directory")
endif()

# Find MLIR package
find_package(MLIR REQUIRED CONFIG PATHS ${MLIR_DIR} NO_DEFAULT_PATH)

if(NOT MLIR_FOUND)
  message(FATAL_ERROR "MLIR not found in ${MLIR_DIR}")
endif()

message(STATUS "Found MLIR ${MLIR_PACKAGE_VERSION}")
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "MLIR Include Dir: ${MLIR_INCLUDE_DIRS}")
message(STATUS "MLIR Binary Dir: ${MLIR_BINARY_DIR}")

# Set MLIR-related variables
set(MLIR_MAIN_SRC_DIR ${MLIR_SOURCE_DIR} CACHE STRING "Location of MLIR source")
set(MLIR_CMAKE_DIR ${MLIR_DIR} CACHE STRING "Location of MLIR CMake modules")
set(MLIR_TABLEGEN_EXE mlir-tblgen)

# If MLIR was built as part of LLVM, set additional paths
if(DEFINED MLIR_TABLEGEN_OUTPUT_DIR)
  include_directories(${MLIR_TABLEGEN_OUTPUT_DIR})
else()
  set(MLIR_TABLEGEN_OUTPUT_DIR ${MLIR_BINARY_DIR}/include)
  include_directories(${MLIR_TABLEGEN_OUTPUT_DIR})
endif()

# Include MLIR CMake modules
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
include(AddMLIR)

# Add MLIR include directories
include_directories(${MLIR_INCLUDE_DIRS})

# Add MLIR library directories
if(DEFINED MLIR_LIBRARY_DIR)
  link_directories(${MLIR_LIBRARY_DIR})
endif()

print_info("✓ MLIR configured successfully\n" "32")
