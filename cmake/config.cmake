# Build configuration options
# NOTE: set(LLVM_BUILD_DIR path/to/your/pre-built/installation) # if you're using your pre-built installation.
set(LLVM_BUILD_DIR /path/to/your/pre-built/installation)

# Compilation options
set(USE_CCACHE ON)
set(DEFAULT_SYSROOT "") # for macOS user you can use `xcrun --show-sdk-path` to fill in this PATH

# Build options
set(ENABLE_TESTING ON)

set(BUILD_FRONTEND ON) # Build the reusable Clang AST lowering library
set(BUILD_CLANG_TO_MLIR ON) # Set OFF to build libraries without the CLI

# Derived paths
# NOTE: you can force cmake to overwrite these PATHs
set(LLVM_DIR ${LLVM_BUILD_DIR}/lib/cmake/llvm)
set(MLIR_DIR ${LLVM_BUILD_DIR}/lib/cmake/mlir)
set(Clang_DIR ${LLVM_BUILD_DIR}/lib/cmake/clang)
