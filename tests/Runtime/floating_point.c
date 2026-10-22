// RUN: clang-to-mlir %s -o %t.mlir -- -ffast-math
// RUN: mlir-opt --convert-vector-to-llvm --convert-scf-to-cf --convert-to-llvm --reconcile-unrealized-casts %t.mlir -o %t.llvm.mlir
// RUN: mlir-translate --mlir-to-llvmir %t.llvm.mlir -o %t.ll
// RUN: lli %t.ll | FileCheck %s
// RUN: clang-to-mlir --disable-opt %s -o %t.raw.mlir -- -ffast-math
// RUN: mlir-opt --expand-strided-metadata --convert-scf-to-cf --convert-to-llvm %t.raw.mlir -o %t.raw.llvm.mlir
// RUN: mlir-translate --mlir-to-llvmir %t.raw.llvm.mlir -o %t.raw.ll
// RUN: lli %t.raw.ll | FileCheck %s
// RUN: FileCheck %s --check-prefix=VECTOR --input-file=%t.mlir

#include <stdio.h>

float sum3(const float *a, int n, float initial) {
  float result = initial;
#pragma cmlir loop vectorize(enable) vectorize_width(3)
  for (int i = 3; i < n; ++i) {
    result += a[i];
  }
  return result;
}

// VECTOR-LABEL: func.func @sum3(
// VECTOR: vector.create_mask
// VECTOR: vector.reduction <add>

int main(void) {
  float values[23];
  for (int i = 0; i < 23; ++i) {
    values[i] = i + 1;
  }
  int errors = 0;
  // These small integer-valued sums are exact in f32 in either grouping.
  for (int n = -1; n <= 19; ++n) {
    float expected = 7;
    for (int i = 3; i < n; ++i) {
      expected += values[i];
    }
    if (sum3(values, n, 7) != expected) {
      ++errors;
    }
  }
  printf("floating-point reductions: %d\n", errors);
  return errors;
}

// CHECK: floating-point reductions: 0
