// RUN: not clang-to-mlir --disable-opt %s -- -ffp-model=strict 2>&1 | FileCheck %s
// RUN: not clang-to-mlir --disable-opt %s -- -frounding-math 2>&1 | FileCheck %s

float add(float a, float b) { return a + b; }

// CHECK: constrained floating-point expressions are unsupported
