// RUN: cmlirc %s -function=main -o %t.input.mlir
// RUN: mlir-opt --convert-scf-to-cf --convert-to-llvm %t.input.mlir -o %t.llvm.mlir
// RUN: mlir-translate --mlir-to-llvmir %t.llvm.mlir -o %t.ll
// RUN: lli %t.ll | FileCheck %s

#include <stdio.h>

int main() {
  for (int i = 0; i < 10; i++) {
    if (i == 3) {
      break;
    }
    printf("i is %d\n", i);
  }

  printf("Loop done!");

  return 0;
}

// CHECK: i is 0
// CHECK: i is 1
// CHECK: i is 2
// CHECK: Loop done!
