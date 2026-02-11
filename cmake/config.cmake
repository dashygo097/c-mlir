# Build configuration options
# NOTE: set(LLVM_BUILD_DIR path/to/your/pre-built/installation) # if you're using your pre-built installation.
set(LLVM_BUILD_DIR ${CMAKE_SOURCE_DIR}/llvm-project/build) # If you're building this proj from scratch, you should not change it

# Compilation options
set(USE_CCACHE ON)

# Build options
set(BUILD_EXECUTABLES ON)
set(ENABLE_TESTING OFF)

# Derived paths
# NOTE: you can force cmake to overwrite these PATHs
set(LLVM_DIR ${LLVM_BUILD_DIR}/lib/cmake/llvm)
set(MLIR_DIR ${LLVM_BUILD_DIR}/lib/cmake/mlir)
set(Clang_DIR ${LLVM_BUILD_DIR}/lib/cmake/clang)
