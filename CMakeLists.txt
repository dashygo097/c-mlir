cmake_minimum_required(VERSION 3.24)
project(cmlir LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Add cmake module path
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

include(Helpers)
include(CompilerSetup)
include(Platforms)

# Generate and include build configuration
if(EXISTS "${CMAKE_BINARY_DIR}/config.cmake")
  include("${CMAKE_BINARY_DIR}/config.cmake")
endif()

# Add cmake modules
include(AppendLLVM)
include(AppendMLIR)
include(AppendClang)
include(AppendCIRCT)

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

if(BUILD_EXECUTABLES)
  add_subdirectory(tools)
endif()

# Tests
if(BUILD_EXECUTABLES AND ENABLE_TESTING)
  add_subdirectory(tests)
endif()
