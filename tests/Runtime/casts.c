// RUN: clang-to-mlir %s -function=main -o %t.mlir -- -xc
// RUN: mlir-opt --convert-to-llvm %t.mlir -o %t.llvm.mlir
// RUN: mlir-translate --mlir-to-llvmir %t.llvm.mlir -o %t.ll
// RUN: lli %t.ll | FileCheck %s
// RUN: clang-to-mlir %s -function=main -disable-opt -o %t.raw.mlir -- -xc
// RUN: mlir-opt --convert-to-llvm %t.raw.mlir -o %t.raw.llvm.mlir
// RUN: mlir-translate --mlir-to-llvmir %t.raw.llvm.mlir -o %t.raw.ll
// RUN: lli %t.raw.ll | FileCheck %s

#include <stdio.h>

int lossy(int value) { return (int)(float)value; }

int exact(int value) { return (int)(double)value; }

int fractional(int value) { return (int)((float)value + 0.75f); }

unsigned unsigned_lossy(unsigned value) { return (unsigned)(float)value; }

long long_lossy(long value) { return (long)(double)value; }

unsigned long unsigned_long_lossy(unsigned long value) {
  return (unsigned long)(double)value;
}

int main() {
  printf("casts %d %d %d %u\n", lossy(16777217), exact(16777217),
         fractional(-1), unsigned_lossy(16777217));
  printf("wide %ld %lu\n", long_lossy(9007199254740993L),
         unsigned_long_lossy(9007199254740993UL));
  return 0;
}

// CHECK: casts 16777216 16777217 0 16777216
// CHECK: wide 9007199254740992 9007199254740992
