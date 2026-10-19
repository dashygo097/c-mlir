// RUN: clang-to-mlir --disable-opt %s --function=main -o %t.raw.mlir -- -xc
// RUN: mlir-opt --expand-strided-metadata --convert-scf-to-cf --convert-to-llvm %t.raw.mlir -o %t.raw.llvm.mlir
// RUN: mlir-translate --mlir-to-llvmir %t.raw.llvm.mlir -o %t.raw.ll
// RUN: lli %t.raw.ll | FileCheck %s
// RUN: clang-to-mlir %s --function=main -o %t.mlir -- -xc
// RUN: mlir-opt --expand-strided-metadata --convert-scf-to-cf --convert-to-llvm %t.mlir -o %t.llvm.mlir
// RUN: mlir-translate --mlir-to-llvmir %t.llvm.mlir -o %t.ll
// RUN: lli %t.ll | FileCheck %s

#include <stdio.h>

void for_continue(int n, int *out) {
  for (int i = 0; i < n; ++i) {
    if (i == 2) {
      continue;
    }
    out[i] = i;
  }
}

void while_continue(int n, int *out) {
  int i = 0;
  while (i < n) {
    ++i;
    if (i == 2) {
      continue;
    }
    out[i] = i;
  }
}

void do_continue(int n, int *out) {
  int i = 0;
  do {
    ++i;
    if (i == 2) {
      continue;
    }
    out[i] = i;
  } while (i < n);
}

int nested_continue(int n) {
  int count = 0;
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      if (j == 2) {
        continue;
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
  for_continue(5, out);
  printf("for %d %d %d\n", out[1], out[2], out[4]);
  reset(out);
  while_continue(5, out);
  printf("while %d %d %d\n", out[1], out[2], out[5]);
  reset(out);
  do_continue(5, out);
  printf("do %d %d %d\n", out[1], out[2], out[5]);
  reset(out);
  do_continue(0, out);
  printf("once %d\n", out[1]);
  printf("nested %d\n", nested_continue(4));
  return 0;
}

// CHECK: for 1 -1 4
// CHECK-NEXT: while 1 -1 5
// CHECK-NEXT: do 1 -1 5
// CHECK-NEXT: once 1
// CHECK-NEXT: nested 12
