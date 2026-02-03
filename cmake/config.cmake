# Build configuration options
set(LLVM_BUILD_DIR ${CMAKE_SOURCE_DIR}/llvm-project/build)

# Compilation options
set(USE_CCACHE ON)

# Build options
set(BUILD_EXECUTABLES ON)
set(ENABLE_TESTING OFF)

# Derived paths
set(LLVM_DIR ${LLVM_BUILD_DIR}/lib/cmake/llvm)
set(MLIR_DIR ${LLVM_BUILD_DIR}/lib/cmake/mlir)
set(Clang_DIR ${LLVM_BUILD_DIR}/lib/cmake/clang)
