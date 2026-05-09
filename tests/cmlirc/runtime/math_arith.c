// RUN: cmlirc %s -function=main -o %t.input.mlir
// RUN: mlir-opt --convert-math-to-llvm --convert-to-llvm %t.input.mlir -o %t.llvm.mlir
// RUN: mlir-translate --mlir-to-llvmir %t.llvm.mlir -o %t.ll
// RUN: lli %t.ll | FileCheck %s

#include <math.h>
#include <stdio.h>

int main() {
  double a = fabs(-2.5);
  double b = sqrt(16.0);
  double c = floor(3.8);
  double d = ceil(3.2);
  double e = trunc(5.9);

  printf("%.1f %.1f %.1f %.1f %.1f\n", a, b, c, d, e);

  return 0;
}

// CHECK: 2.5 4.0 3.0 4.0 5.0
