# c-mlir

## Prerequisites

- `C/C++ toolchains` (compiler, linker etc.)
- `cmake`
- `make` or `ninja`.

## Installation

### Option 1: Build from scratch

**1. Configuration:**

```bash
make config # generate configuration file at build/config.cmake
```

**2. Build LLVM, MLIR and Clang:**

```bash
mkdir llvm-project/build
cd llvm-project/build
cmake -G Ninja ../llvm \
  -DLLVM_ENABLE_PROJECTS="mlir;clang" \
  -DLLVM_TARGETS_TO_BUILD="host" \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DCMAKE_BUILD_TYPE=DEBUG # Or `Release` etc.
ninja
ninja check-mlir
```

For faster compilation we recommend using -DLLVM_USE_LINKER=lld.

**3. Build `cmlirc`:**

```bash
make
```

Or manually follow this:

```bash
mkdir build
cd build
cmake -G Ninja ..
ninja
```

### Option 2: Build with pre-built LLVM, MLIR and Clang

**1. Configuration:**

```bash
make config # generate configuration file at build/config.cmake
```

Specify the `PATH` in `config.cmake`.

```cmake
# Build configuration options
# NOTE: set(LLVM_BUILD_DIR path/to/your/pre-built/installation) # if you're using your pre-built installation.
set(LLVM_BUILD_DIR ${CMAKE_SOURCE_DIR}/llvm-project/build) # If you're building this proj from scratch, you should not change it

# Compilation options
set(USE_CCACHE ON)

# Build options
set(BUILD_EXECUTABLES ON)
set(ENABLE_TESTING ON)

# Derived paths
# NOTE: you can force cmake to overwrite these PATHs
set(LLVM_DIR ${LLVM_BUILD_DIR}/lib/cmake/llvm)
set(MLIR_DIR ${LLVM_BUILD_DIR}/lib/cmake/mlir)
set(Clang_DIR ${LLVM_BUILD_DIR}/lib/cmake/clang)
```

**2. Build `cmlirc`:**

```bash
make
```

Or manually follow this:

```bash
mkdir build
cd build
cmake -G Ninja ..
ninja
```

## Acknowledgement

- [Polygeist](https://github.com/llvm/Polygeist.git)

## License

Copyright 2026 dashygo097

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.
