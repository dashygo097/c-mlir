// RUN: clang-to-mlir --disable-opt %s --function=main -o %t.raw.mlir -- -xc
// RUN: mlir-opt --expand-strided-metadata --convert-scf-to-cf --convert-to-llvm %t.raw.mlir -o %t.raw.llvm.mlir
// RUN: mlir-translate --mlir-to-llvmir %t.raw.llvm.mlir -o %t.raw.ll
// RUN: lli %t.raw.ll | FileCheck %s
// RUN: clang-to-mlir %s --function=main -o %t.mlir -- -xc
// RUN: mlir-opt --expand-strided-metadata --convert-scf-to-cf --convert-to-llvm %t.mlir -o %t.llvm.mlir
// RUN: mlir-translate --mlir-to-llvmir %t.llvm.mlir -o %t.ll
// RUN: lli %t.ll | FileCheck %s

#include <stdio.h>

void for_break(int n, int *out) {
  for (int i = 0; i < n; ++i) {
    if (i == 2) {
      break;
    }
    out[i] = i;
  }
}

void while_break(int n, int *out) {
  int i = 0;
  while (i < n) {
    if (i == 2) {
      break;
    }
    out[i] = i;
    ++i;
  }
}

void do_break(int n, int *out) {
  int i = 0;
  do {
    if (i == 2) {
      break;
    }
    out[i] = i;
    ++i;
  } while (i < n);
}

int nested_break(int n) {
  int count = 0;
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      if (j == 2) {
        break;
      }
      ++count;
    }
  }
  return count;
}

void reset(int *out) {
  for (int i = 0; i < 8; ++i) {
    out[i] = -1;
  }
}

int main(void) {
  int out[8];
  reset(out);
  for_break(5, out);
  printf("for %d %d %d\n", out[0], out[1], out[2]);
  reset(out);
  while_break(5, out);
  printf("while %d %d %d\n", out[0], out[1], out[2]);
  reset(out);
  do_break(5, out);
  printf("do %d %d %d\n", out[0], out[1], out[2]);
  reset(out);
  do_break(0, out);
  printf("once %d %d\n", out[0], out[1]);
  printf("nested %d\n", nested_break(4));
  return 0;
}

// CHECK: for 0 1 -1
// CHECK-NEXT: while 0 1 -1
// CHECK-NEXT: do 0 1 -1
// CHECK-NEXT: once 0 -1
// CHECK-NEXT: nested 8
