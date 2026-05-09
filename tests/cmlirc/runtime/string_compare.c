// RUN: cmlirc %s -function=main -o %t.input.mlir
// RUN: mlir-opt --convert-to-llvm %t.input.mlir -o %t.llvm.mlir
// RUN: mlir-translate --mlir-to-llvmir %t.llvm.mlir -o %t.ll
// RUN: lli %t.ll | FileCheck %s

#include <stdio.h>
#include <string.h>

int main() {
  int len = (int)strlen("compiler");
  int eq = strcmp("abc", "abc") == 0 ? 1 : 0;
  int lt = strcmp("abc", "abd") < 0 ? 1 : 0;
  int prefix = strncmp("abcdef", "abcxyz", 3) == 0 ? 1 : 0;

  printf("%d %d %d %d\n", len, eq, lt, prefix);

  return 0;
}

// CHECK: 8 1 1 1
