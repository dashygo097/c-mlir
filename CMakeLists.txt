cmake_minimum_required(VERSION 3.24)
project(cmlir LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Add cmake module path
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

include(Helpers)
include(CompilerSetup)
include(Platforms)

# Generate and include build configuration
if(EXISTS "${CMAKE_BINARY_DIR}/config.cmake")
  include("${CMAKE_BINARY_DIR}/config.cmake")
endif()

# Add cmake modules
set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)

include(AppendLLVM)
include(AppendMLIR)
include(AppendClang)

# Build the library
add_subdirectory(include)
add_subdirectory(lib)

# Executables
if(BUILD_EXECUTABLES)
  add_subdirectory(tools)
endif()

# Tests
if(ENABLE_TESTING)
  enable_testing()
  add_subdirectory(tests)
endif()
