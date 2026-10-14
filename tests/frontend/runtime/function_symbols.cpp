// RUN: clang-to-mlir %s -function=main -o %t.mlir -- -std=c++17
// RUN: mlir-opt --convert-scf-to-cf --convert-to-llvm %t.mlir -o %t.llvm.mlir
// RUN: mlir-translate --mlir-to-llvmir %t.llvm.mlir -o %t.ll
// RUN: lli %t.ll | FileCheck %s
// RUN: clang-to-mlir %s -o %t.all.mlir -- -std=c++17
// RUN: mlir-opt %t.all.mlir -o %t.verified.mlir
#include <stdio.h>
extern "C" int getpid();
int forward(int);

int forward(int n) { return n + 1; }

int overloaded(int n) { return n + 2; }

int overloaded(int a, int b) { return a + b; }

namespace first {
int value() { return 3; }
} // namespace first

namespace second {
int value() { return 4; }
} // namespace second

int recursive(int n) {
  if (n <= 0) {
    return 0;
  }
  return 1 + recursive(n - 1);
}

int main() {
  printf("symbols %d %d %d %d %d %d\n", forward(2), overloaded(2),
         overloaded(2, 3), first::value(), second::value(), recursive(3));
  printf("external %d\n", getpid() > 0);
  return 0;
}

// CHECK: symbols 3 4 5 3 4 3
// CHECK: external 1
