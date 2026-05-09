// RUN: cmlirc %s -function=main -o %t.input.mlir
// RUN: mlir-opt --convert-math-to-llvm --convert-to-llvm %t.input.mlir -o %t.llvm.mlir
// RUN: mlir-translate --mlir-to-llvmir %t.llvm.mlir -o %t.ll
// RUN: lli %t.ll | FileCheck %s

#include <math.h>
#include <stdio.h>

int main() {
  double a = sin(0.0);
  double b = cos(0.0);
  double c = tan(0.0);
  double d = exp(0.0);
  double e = log(1.0);

  printf("%.1f %.1f %.1f %.1f %.1f\n", a, b, c, d, e);

  return 0;
}

// CHECK: 0.0 1.0 0.0 1.0 0.0
