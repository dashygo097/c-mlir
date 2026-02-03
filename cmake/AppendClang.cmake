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

# Common Clang libraries needed for most projects
set(CLANG_LIBS
  clangBasic
  clangLex
  clangParse
  clangAST
  clangSema
  clangCodeGen
  clangAnalysis
  clangEdit
  clangRewrite
  clangDriver
  clangSerialization
  clangFrontend
  clangFrontendTool
  clangTooling
  clangToolingCore
)

# Function to add Clang-based executable
function(add_clang_executable name)
  add_executable(${name} ${ARGN})
  
  # Link against Clang libraries
  target_link_libraries(${name} PRIVATE ${CLANG_LIBS})
  
  set_target_properties(${name} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  )
endfunction()

# Function to add Clang-based library
function(add_clang_library name)
  add_library(${name} ${ARGN})
  
  # Link against Clang libraries
  target_link_libraries(${name} PUBLIC ${CLANG_LIBS})
  
  set_target_properties(${name} PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
  )
  
  # Add include directories
  target_include_directories(${name} PUBLIC
    ${CLANG_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_BINARY_DIR}/include
  )
endfunction()

# Function to add a Clang tool (combines Clang and LLVM)
function(add_clang_tool name)
  add_clang_executable(${name} ${ARGN})
  
  # Also link LLVM support
  target_link_libraries(${name} PRIVATE
    LLVMSupport
    LLVMCore
  )
endfunction()

print_info("✓ Clang configured successfully\n" "32")
