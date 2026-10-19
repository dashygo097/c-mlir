// RUN: clang-to-mlir --disable-opt %s -- | FileCheck %s --enable-var-scope

int integer(int value, int rhs) {
  value += rhs;
  value -= rhs;
  value *= rhs;
  value /= rhs;
  value %= rhs;
  return value;
}

// CHECK-LABEL: func.func @integer
// CHECK: arith.addi
// CHECK: memref.store
// CHECK: arith.subi
// CHECK: memref.store
// CHECK: arith.muli
// CHECK: arith.divsi
// CHECK: arith.remsi

int bits(int value, int rhs) {
  value &= rhs;
  value |= rhs;
  value ^= rhs;
  value <<= 1;
  value >>= 1;
  return value;
}

// CHECK-LABEL: func.func @bits
// CHECK: arith.andi
// CHECK: arith.ori
// CHECK: arith.xori
// CHECK: arith.shli
// CHECK: arith.shrsi

float floating(float value, float rhs) {
  value += rhs;
  value *= rhs;
  return value;
}

// CHECK-LABEL: func.func @floating
// CHECK: arith.addf
// CHECK: arith.mulf
