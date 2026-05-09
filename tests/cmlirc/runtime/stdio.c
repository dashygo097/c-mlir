// RUN: cmlirc %s -function=main -o %t.input.mlir
// RUN: mlir-opt --convert-to-llvm %t.input.mlir -o %t.llvm.mlir
// RUN: mlir-translate --mlir-to-llvmir %t.llvm.mlir -o %t.ll
// RUN: lli %t.ll | FileCheck %s

#include <stdio.h>

int main() {
  printf("hello ");
  puts("stdio");

  putchar('O');
  putchar('K');
  putchar('\n');

  return 0;
}

// CHECK: hello stdio
// CHECK-NEXT: OK
