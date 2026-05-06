# c-mlir

## Prerequisites

- `C/C++ toolchains` (compiler, linker etc.)
- `cmake`
- `make` or `ninja`.
- `lit`(optional for testing)

## Installation

### Option 1: Build from scratch

**1. Configuration:**

```bash
make config # generate configuration file at build/config.cmake
```

**2. Build LLVM, MLIR and Clang:**

```bash
mkdir circt/llvm/build
cd circt/llvm/build
cmake -G Ninja ../llvm \
  -DLLVM_ENABLE_PROJECTS="mlir;clang" \
  -DLLVM_TARGETS_TO_BUILD="host" \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DCMAKE_BUILD_TYPE=DEBUG
ninja
ninja check-mlir
```

For faster compilation, it is recommended to use -DLLVM_USE_LINKER=lld.

**2.1 Build CIRCT:(_optional_)**

```bash
mkdir circt/build
cd circt/build
cmake -G Ninja .. \
    -DMLIR_DIR=$PWD/../llvm/build/lib/cmake/mlir \
    -DLLVM_DIR=$PWD/../llvm/build/lib/cmake/llvm \
    -DLLVM_ENABLE_ASSERTIONS=ON \
    -DCMAKE_BUILD_TYPE=DEBUG \
    -DLLVM_USE_SPLIT_DWARF=ON
ninja
```

For faster compilation, it is recommended to use -DLLVM_USE_LINKER=lld.

**3. Build tools:**

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
set(LLVM_BUILD_DIR ${CMAKE_SOURCE_DIR}/circt/llvm/build) # If you're building this proj from scratch, you should not change it
set(CIRCT_BUILD_DIR ${CMAKE_SOURCE_DIR}/circt/build) # If you're not building `chwc`(c-hardware compiler), you can ignore this

# Compilation options
set(USE_CCACHE ON)

# Build options
set(BUILD_EXECUTABLES ON)
set(ENABLE_TESTING ON)

set(ENABLE_CMLIRC ON) # Set `OFF` to disable building tool `cmlirc`
set(ENABLE_CHWC ON) # Set `OFF` to disable building tool `chwc`

# Derived paths
# NOTE: you can force cmake to overwrite these PATHs
set(LLVM_DIR ${LLVM_BUILD_DIR}/lib/cmake/llvm)
set(MLIR_DIR ${LLVM_BUILD_DIR}/lib/cmake/mlir)
set(Clang_DIR ${LLVM_BUILD_DIR}/lib/cmake/clang)
set(CIRCT_DIR ${CIRCT_BUILD_DIR}/lib/cmake/circt) # If you're not building `chwc`(c-hardware compiler), you can ignore this
```

**2. Build tools**

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

## Run Tests

**1. Install `lit`**

Activate a python virtual env(`venv`, `conda`, `uv` etc.) and install `lit` using `pip`:

```bash
pip install lit
```

**2. Run tests**

```bash
make test
```

Or manually follow this:

```bash
cd build
ninja check-cmlirc
```

## Examples

### `cmlirc` (c/c++ to mlir compiler)

**1. Dot Product**

```c++
float dot(const float *a, const float *b, int n) {
  float result = 0;
  for (int i = 0; i < n; i++) {
    result += a[i] * b[i];
  }
  return result;
}
```

```bash
cmlirc dot.c
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

### `chwc` (c/c++ to hardware compiler)

**1. Counter**

```c++
#include <chwc/Runtime.h> // Runtime Lib

class Counter final : public Hardware {
public:
  Input<UInt<1>> en;
  Output<UInt<16>> out;
  Reg<UInt<16>> value;

  // HW_RESET denotes this function processes reset tasks
  HW_RESET void rst() { value = 0; }

  // HW_CLOCK_TICK denotes this function processes clock related tasks
  HW_CLOCK_TICK void tick() {
    if (en) {
      value = add_one(value);
    }
    out = value;
  }

  // HW_FUNC denotes this function is an inline helper function
  HW_FUNC UInt<16> add_one(UInt<16> input) { return input + 1; }
};
```

```bash
chwc counter.cpp
```

```mlir
module {
  hw.module @Counter(in %clk : !seq.clock, in %rst : i1, in %en : i1, out out : i16) {
    %c1_i16 = arith.constant 1 : i16
    %c0_i16 = arith.constant 0 : i16
    %value = seq.firreg %1 clock %clk reset sync %rst, %c0_i16 : i16
    %0 = comb.add %value, %c1_i16 : i16
    %1 = comb.mux %en, %0, %value : i16
    hw.output %value : i16
  }
}
```

which can be lowered into (system)verilog with `circt-opt` and `firtool`.

## Acknowledgement

- This work is inspired by [Polygeist](https://github.com/llvm/Polygeist.git)

## License

Copyright 2026 dashygo097

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.
