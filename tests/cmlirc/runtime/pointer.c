// RUN: cmlirc %s -function=main -o %t.input.mlir
// RUN: mlir-opt --convert-to-llvm %t.input.mlir -o %t.llvm.mlir
// RUN: mlir-translate --mlir-to-llvmir %t.llvm.mlir -o %t.ll
// RUN: lli %t.ll | FileCheck %s

#include <stdio.h>

int main() {
  int a = 123;
  printf("The value of a is %d\n", *(&a));
  return 0;
}

// CHECK: The value of a is 123
