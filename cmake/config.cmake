# Build configuration options
# NOTE: set(LLVM_BUILD_DIR path/to/your/pre-built/installation) # if you're using your pre-built installation.
set(LLVM_BUILD_DIR /path/to/your/pre-built/installation)

# Compilation options
option(USE_CCACHE "Enable ccache for incremental building" ON)
set(DEFAULT_SYSROOT "") # for macOS user you can use `xcrun --show-sdk-path` to fill in this PATH

# Build options
option(ENABLE_TESTING "Enable building with tests" ON)
option(ENABLE_EXECUTABLES "Enable building CLI executables" ON) # Build the reusable Clang AST lowering library

# Derived paths
# NOTE: you can force cmake to overwrite these PATHs
set(LLVM_DIR ${LLVM_BUILD_DIR}/lib/cmake/llvm)
set(MLIR_DIR ${LLVM_BUILD_DIR}/lib/cmake/mlir)
set(Clang_DIR ${LLVM_BUILD_DIR}/lib/cmake/clang)
