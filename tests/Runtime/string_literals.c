// RUN: clang-to-mlir --disable-opt %s -function=main -o %t.mlir --
// RUN: mlir-opt --convert-to-llvm %t.mlir -o %t.llvm.mlir
// RUN: mlir-translate --mlir-to-llvmir %t.llvm.mlir -o %t.ll
// RUN: lli %t.ll | FileCheck %s

#include <stdio.h>

int main() {
  printf("Hello World!");
  return 0;
}

// CHECK: Hello World!
