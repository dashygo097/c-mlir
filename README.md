# c-mlir

## Prerequisites

- `C/C++ toolchains` (compiler, linker etc.)
- `cmake`
- `make` or `ninja`.
- `lit`(optional for testing)

## Installation

**1. Configuration:**

```bash
make config # generate configuration file at build/config.cmake
```

Specify the `PATH` in `config.cmake`.

```cmake
# Build configuration options
# NOTE: set(LLVM_BUILD_DIR path/to/your/pre-built/installation) # if you're using your pre-built installation.
set(LLVM_BUILD_DIR /path/to/your/pre-built/installation)

# Compilation options
set(USE_CCACHE ON)
set(DEFAULT_SYSROOT "") # for macOS user you can use `xcrun --show-sdk-path` to fill in this PATH

# Test options
set(ENABLE_TESTING ON)

# Build options
set(BUILD_FRONTEND ON) # Build the reusable Clang AST lowering library
set(BUILD_CLANG_TO_MLIR ON) # Set OFF to build libraries without the CLI

# Derived paths
# NOTE: you can force cmake to overwrite these PATHs
set(LLVM_DIR ${LLVM_BUILD_DIR}/lib/cmake/llvm)
set(MLIR_DIR ${LLVM_BUILD_DIR}/lib/cmake/mlir)
set(Clang_DIR ${LLVM_BUILD_DIR}/lib/cmake/clang)
```

**2. Build tools**

```bash
make
```

Or manually follow this:

```bash
mkdir -p build
cd build
cmake -G Ninja ..
ninja
```

## Run Tests

**1. Install `lit`**

Activate a python virtual env(`venv`, `conda`, `uv` etc.) and install `lit` using `pip`:

```bash
pip install lit
```

**2. Run tests**

```bash
make test # Run the clang-to-mlir regression suite and compiler API checks
```

Or manually follow this:

```bash
cd build
ninja check-frontend
ninja check-frontend-api # Run only the reusable compiler API checks
```

## Examples

### `clang-to-mlir` (c/c++ to mlir compiler)

**1. Dot Product**

```cpp
float dot(const float *a, const float *b, int n) {
  float result = 0;
  for (int i = 0; i < n; i++) {
    result += a[i] * b[i];
  }
  return result;
}
```

```bash
clang-to-mlir dot.c
```

```mlir
module {
  func.func @dot(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: i32) -> f32 {
    %c1 = arith.constant 1 : index
    %cst = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %0 = arith.index_cast %arg2 : i32 to index
    %1 = scf.for %arg3 = %c0 to %0 step %c1 iter_args(%arg4 = %cst) -> (f32) {
      %2 = memref.load %arg0[%arg3] : memref<?xf32>
      %3 = memref.load %arg1[%arg3] : memref<?xf32>
      %4 = arith.mulf %2, %3 : f32
      %5 = arith.addf %arg4, %4 : f32
      scf.yield %5 : f32
    }
    return %1 : f32
  }
}
```

## Acknowledgement

- This work is inspired by [Polygeist](https://github.com/llvm/Polygeist.git)

## License

Copyright 2026 dashygo097

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.
