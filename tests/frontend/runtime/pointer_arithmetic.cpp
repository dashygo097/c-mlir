// RUN: clang-to-mlir %s -function=main -o %t.mlir -- -std=c++17
// RUN: mlir-opt --expand-strided-metadata --convert-scf-to-cf --convert-to-llvm %t.mlir -o %t.llvm.mlir
// RUN: mlir-translate --mlir-to-llvmir %t.llvm.mlir -o %t.ll
// RUN: lli %t.ll | FileCheck %s
// RUN: clang-to-mlir %s -function=main -disable-opt -o %t.raw.mlir -- -std=c++17
// RUN: mlir-opt --expand-strided-metadata --convert-scf-to-cf --convert-to-llvm %t.raw.mlir -o %t.raw.llvm.mlir
// RUN: mlir-translate --mlir-to-llvmir %t.raw.llvm.mlir -o %t.raw.ll
// RUN: lli %t.raw.ll | FileCheck %s

#include <cstdio>
#include <cstdlib>

int next(int *count) { return ++*count; }

int view(int *pointer, int offset) {
  *(pointer + offset) = 17;
  return *((pointer + offset + 1) - 1);
}

int main() {
  int *heap = (int *)malloc(sizeof(int) * 1024);
  if (!heap) {
    return 1;
  }
  *(heap + 0) = 0;
  *(heap + 1) = 1;
  *(heap + 2) = 2;
  printf("heap %d %d %d\n", *heap, *(1 + heap), *(heap + 2));
  int *end = heap + 2;
  signed char negative = -1;
  *(end + negative) = 7;
  unsigned char positive = 255;
  *(heap + positive) = 77;
  printf("offsets %d %d %d\n", *(end - 1), heap[255], *(end + negative));
  int count = 0;
  *(heap + next(&count)) = 9;
  printf("evaluation %d %d\n", count, heap[1]);

  double *wide = (double *)malloc(sizeof(double) * 3);
  if (!wide) {
    free(heap);
    return 1;
  }
  *(wide + 0) = 1.5;
  *(wide + 1) = 6.25;
  printf("scaling %.2f %.2f\n", *wide, wide[1]);
  int local[5] = {0, 0, 0, 0, 0};
  int result = view(local, 2);
  printf("memref %d %d\n", result, local[2]);
  free(wide);
  free(heap);
  return 0;
}

// CHECK: heap 0 1 2
// CHECK: offsets 7 77 7
// CHECK: evaluation 1 9
// CHECK: scaling 1.50 6.25
// CHECK: memref 17 17
