// RUN: cmlirc %s -function=main -o hello_world_input.mlir
// RUN: mlir-opt --convert-to-llvm ./hello_world_input.mlir -o hello_world_output.mlir
// RUN: mlir-translate --mlir-to-llvmir ./hello_world_output.mlir -o hello_world.ll
// RUN: lli hello_world.ll | FileCheck %s

#include <stdio.h>

int main() {
  printf("Hello World!");
  return 0;
}

// CHECK: Hello World!
