# Build configuration options
# NOTE: set(LLVM_BUILD_DIR path/to/your/pre-built/installation) # if you're using your pre-built installation.
set(LLVM_BUILD_DIR ${CMAKE_SOURCE_DIR}/circt/llvm/build) # If you're building this proj from scratch, you should not change it
set(CIRCT_BUILD_DIR ${CMAKE_SOURCE_DIR}/circt/build) # If you're not building `chwc`(c-hardware compiler), you can ignore this

# Compilation options
set(USE_CCACHE ON)
set(DEFAULT_SYSROOT "")

# Build options
set(BUILD_EXECUTABLES ON)
set(ENABLE_TESTING ON)

set(BUILD_CMLIRC ON) # Set `OFF` to disable building tool `cmlirc`
set(BUILD_CHWC ON) # Set `OFF` to disable building tool `chwc`  

# Derived paths
# NOTE: you can force cmake to overwrite these PATHs
set(LLVM_DIR ${LLVM_BUILD_DIR}/lib/cmake/llvm)
set(MLIR_DIR ${LLVM_BUILD_DIR}/lib/cmake/mlir)
set(Clang_DIR ${LLVM_BUILD_DIR}/lib/cmake/clang)
set(CIRCT_DIR ${CIRCT_BUILD_DIR}/lib/cmake/circt) # If you're not building `chwc`(c-hardware compiler), you can ignore this
