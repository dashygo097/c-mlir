// RUN: cmlirc %s -function=main -o %t.input.mlir
// RUN: mlir-opt --convert-to-llvm %t.input.mlir -o %t.llvm.mlir
// RUN: mlir-translate --mlir-to-llvmir %t.llvm.mlir -o %t.ll
// RUN: lli %t.ll | FileCheck %s

#include <stdio.h>

void swap(int *a, int *b) {
  int temp;
  temp = *a;
  *a = *b;
  *b = temp;
}

int main() {
  int a = 123, b = 456;
  swap(&a, &b);
  printf("a: %d, b: %d", a, b);
  return 0;
}

// CHECK: a: 456, b: 123
