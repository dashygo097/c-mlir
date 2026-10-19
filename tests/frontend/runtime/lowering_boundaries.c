// RUN: clang-to-mlir %s -function=main -o %t.mlir -- -xc
// RUN: FileCheck %s --check-prefix=STRUCTURED --implicit-check-not=cf.br --implicit-check-not=cf.cond_br --implicit-check-not=cf.switch < %t.mlir
// RUN: mlir-opt --expand-strided-metadata --convert-scf-to-cf --convert-to-llvm %t.mlir -o %t.llvm.mlir
// RUN: mlir-translate --mlir-to-llvmir %t.llvm.mlir -o %t.ll
// RUN: lli %t.ll | FileCheck %s
// RUN: clang-to-mlir %s -function=main --inline -o %t.inline.mlir -- -xc
// RUN: FileCheck %s --check-prefix=INLINE --implicit-check-not=cf.br --implicit-check-not=cf.cond_br --implicit-check-not=cf.switch < %t.inline.mlir
// RUN: mlir-opt --expand-strided-metadata --convert-scf-to-cf --convert-to-llvm %t.inline.mlir -o %t.inline.llvm.mlir
// RUN: mlir-translate --mlir-to-llvmir %t.inline.llvm.mlir -o %t.inline.ll
// RUN: lli %t.inline.ll | FileCheck %s
// RUN: clang-to-mlir %s -function=main -disable-opt -o %t.raw.mlir -- -xc
// RUN: mlir-opt --expand-strided-metadata --convert-scf-to-cf --convert-to-llvm %t.raw.mlir -o %t.raw.llvm.mlir
// RUN: mlir-translate --mlir-to-llvmir %t.raw.llvm.mlir -o %t.raw.ll
// RUN: lli %t.raw.ll | FileCheck %s
#include <stdio.h>

// STRUCTURED-LABEL: func.func @loop_return
// STRUCTURED: scf.while
// STRUCTURED: scf.if
// STRUCTURED-LABEL: func.func @nested
// STRUCTURED: scf.while

// INLINE-LABEL: func.func @main
// INLINE: return

int short_circuit(int flag) {
  int x = 0;
  int a = flag && ++x;
  int b = !flag || ++x;
  return x + a + b;
}

int loop_return(int n) {
  int i = n;
  while (i > 0) {
    if (i == 2) {
      return 7;
    }
    --i;
  }
  return 9;
}

int parameter(int n) {
  if (n > 0) {
    ++n;
  }
  for (int i = 0; i < 2; ++i) {
    n += 2;
  }
  return n;
}

int indexed(void) {
  int a[2] = {3, 4};
  int b[2] = {1, 0};
  a[b[0]] += a[b[1]];
  int old = a[1]++;
  return old + a[1];
}

int nested(void) {
  int total = 0;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      if (j == 1) {
        break;
      }
      total += 1;
    }
    if (i == 1) {
      continue;
    }
    total += 10;
  }
  return total;
}

int switch_flow(int n) {
  int result = 0;
  switch (n) {
  case 0:
    result += 1;
  case 1:
    result += 2;
  case 2:
    if (n != 0) {
      break;
    }
    result += 4;
  default:
    result += 8;
  }
  return result;
}

int varying_bound(void) {
  int n = 5, sum = 0;
  for (int i = 0; i < n; ++i) {
    sum += i;
    --n;
  }
  return sum;
}

int aliased_bound(void) {
  int n = 5, sum = 0;
  int *pointer = &n;
  for (int i = 0; i < n; ++i) {
    sum += i;
    --pointer[0];
  }
  return sum;
}

int comparison(int a, int b) { return a < b; }

int main(void) {
  printf("logical %d %d\n", short_circuit(0), short_circuit(1));
  printf("return %d %d\n", loop_return(0), loop_return(3));
  printf("parameter %d %d\n", parameter(0), parameter(2));
  printf("indexed %d nested %d\n", indexed(), nested());
  printf("switch %d %d %d\n", switch_flow(0), switch_flow(1), switch_flow(4));
  printf("bound %d compare %d\n", varying_bound(), comparison(1, 2));
  printf("alias %d\n", aliased_bound());
  return 0;
}

// CHECK: logical 1 4
// CHECK: return 9 7
// CHECK: parameter 4 7
// CHECK: indexed 15 nested 23
// CHECK: switch 15 2 8
// CHECK: bound 3 compare 1
// CHECK: alias 3
