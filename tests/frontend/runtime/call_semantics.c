// RUN: clang-to-mlir %s -function=main -o %t.mlir
// RUN: mlir-opt --convert-math-to-llvm --convert-scf-to-cf --convert-to-llvm %t.mlir -o %t.llvm.mlir
// RUN: mlir-translate --mlir-to-llvmir %t.llvm.mlir -o %t.ll
// RUN: lli %t.ll | FileCheck %s
// RUN: clang-to-mlir %s -function=main -disable-opt -o %t.raw.mlir
// RUN: mlir-opt --convert-math-to-llvm --convert-scf-to-cf --convert-to-llvm %t.raw.mlir -o %t.raw.llvm.mlir
// RUN: mlir-translate --mlir-to-llvmir %t.raw.llvm.mlir -o %t.raw.ll
// RUN: lli %t.raw.ll | FileCheck %s

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

int forward(int);

int increment(int *value) { return ++*value; }

int sum(int first, int second) { return first + second; }

void set(int *value, int replacement) { *value = replacement; }

int main() {
  int count = 0;
  int total = sum(increment(&count), increment(&count));
  set(&count, count + 10);
  float remainder = fmodf(5.5f, 2.0f);
  float scaled = ldexpf(1.5f, 2);
  unsigned char small = 250;
  // Varargs must keep default integer and float promotions.
  printf("calls %d %d %d %.1f %.1f %u\n", total, count, forward(3),
         remainder, scaled, small);
  printf("classification %d %d\n", __builtin_isfinite(1.0f),
         __builtin_isnan(1.0));
  // Exercise the existing raw-pointer allocation ABI while the structured
  // allocation representation is developed separately.
  int *heap = (int *)malloc(2 * sizeof(int));
  if (!heap) {
    return 1;
  }
  heap[0] = 7;
  heap[1] = 9;
  printf("heap %d\n", heap[0] + heap[1]);
  free(heap);
  return 0;
}

int forward(int value) { return value + 1; }

// CHECK: calls 3 12 4 1.5 6.0 250
// CHECK: classification 1 0
// CHECK: heap 16
