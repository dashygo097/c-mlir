// RUN: cmlirc %s -function=main -o %t.input.mlir
// RUN: mlir-opt --convert-to-llvm %t.input.mlir -o %t.llvm.mlir
// RUN: mlir-translate --mlir-to-llvmir %t.llvm.mlir -o %t.ll
// RUN: lli %t.ll | FileCheck %s

#include <stdio.h>
#include <stdlib.h>

int main() {
  int a = atoi("123");
  long b = atol("4096");
  double c = atof("2.5");
  int d = abs(-9);

  printf("%d %ld %.1f %d\n", a, b, c, d);

  return 0;
}

// CHECK: 123 4096 2.5 9
