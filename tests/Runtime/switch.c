// RUN: clang-to-mlir --disable-opt %s --function=main -o %t.raw.mlir -- -xc
// RUN: mlir-opt --expand-strided-metadata --convert-scf-to-cf --convert-to-llvm %t.raw.mlir -o %t.raw.llvm.mlir
// RUN: mlir-translate --mlir-to-llvmir %t.raw.llvm.mlir -o %t.raw.ll
// RUN: lli %t.raw.ll | FileCheck %s
// RUN: clang-to-mlir %s --function=main -o %t.mlir -- -xc
// RUN: mlir-opt --expand-strided-metadata --convert-scf-to-cf --convert-to-llvm %t.mlir -o %t.llvm.mlir
// RUN: mlir-translate --mlir-to-llvmir %t.llvm.mlir -o %t.ll
// RUN: lli %t.ll | FileCheck %s

#include <stdio.h>

int fallthrough(int value) {
  int result = 0;
  switch (value) {
  case -1:
    result = 1;
  case 0:
    result += 2;
    break;
  default:
    result = 3;
  }
  return result;
}

int shared_cases(int value) {
  switch (value) {
  case 1:
  case 2:
    return 7;
  default:
    return 9;
  }
}

int no_default(int value) {
  int result = 0;
  switch (value) {
  case 1:
    result = 1;
    break;
  }
  return result;
}

int nested_break(int n) {
  int result = 0;
  for (int i = 0; i < n; ++i) {
    switch (i) {
    case 1:
      break;
    default:
      ++result;
    }
    ++result;
  }
  return result;
}

int main(void) {
  printf("fallthrough %d %d %d\n", fallthrough(-1), fallthrough(0), fallthrough(4));
  printf("shared %d %d %d\n", shared_cases(1), shared_cases(2), shared_cases(3));
  printf("default %d %d\n", no_default(0), no_default(1));
  printf("nested %d\n", nested_break(3));
  return 0;
}

// CHECK: fallthrough 3 2 3
// CHECK-NEXT: shared 7 7 9
// CHECK-NEXT: default 0 1
// CHECK-NEXT: nested 5
