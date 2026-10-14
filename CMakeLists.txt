cmake_minimum_required(VERSION 3.24)
project(cmlir LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Add cmake module path
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/lib)

include(Helpers)
include(CompilerSetup)
include(Platforms)

# Generate and include build configuration
if(EXISTS "${CMAKE_CURRENT_BINARY_DIR}/config.cmake")
  include("${CMAKE_CURRENT_BINARY_DIR}/config.cmake")
endif()

option(BUILD_FRONTEND "Build the Clang AST lowering library" ON)
option(BUILD_CLANG_TO_MLIR "Build the clang-to-mlir command-line tool" ${BUILD_FRONTEND})
if(BUILD_CLANG_TO_MLIR AND NOT BUILD_FRONTEND)
  message(FATAL_ERROR "BUILD_CLANG_TO_MLIR requires BUILD_FRONTEND")
endif()

# Add cmake modules
include(AppendLLVM)
include(AppendMLIR)
include(AppendClang)

# Build the library
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)

add_subdirectory(include)
add_subdirectory(lib)

# Executables
if(NOT DEFINED DEFAULT_SYSROOT)
  set(DEFAULT_SYSROOT "")
endif()
message(STATUS "Default sysroot: ${DEFAULT_SYSROOT}")
add_compile_definitions(DEFAULT_SYSROOT="${DEFAULT_SYSROOT}")
add_subdirectory(tools)

# Tests
if(ENABLE_TESTING)
  enable_testing()
  add_subdirectory(tests)
endif()
