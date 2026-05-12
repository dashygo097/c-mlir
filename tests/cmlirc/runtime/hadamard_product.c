// RUN: cmlirc %s -function=main -o %t.input.mlir
// RUN: mlir-opt --convert-scf-to-cf --convert-to-llvm %t.input.mlir -o %t.llvm.mlir
// RUN: mlir-translate --mlir-to-llvmir %t.llvm.mlir -o %t.ll
// RUN: lli %t.ll | FileCheck %s

#include <stdio.h>

int main() {
  int A[2][2], B[2][2], C[2][2];

  for (int i = 0; i < 2; i++) {
    for (int j = 1; j >= 0; j--) {
      A[i][j] = i + j;
      B[i][j] = i + j;
    }
  }

  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < 2; j++) {
      C[i][j] = A[i][j] * B[i][j];
      printf("C[%d][%d] = %d\n", i, j, C[i][j]);
    }
  }

  return 0;
}

// CHECK: C[0][0] = 0
// CHECK: C[0][1] = 1
// CHECK: C[1][0] = 1
// CHECK: C[1][1] = 4
