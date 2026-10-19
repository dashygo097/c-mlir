// RUN: clang-to-mlir --disable-opt %s -- | FileCheck %s --enable-var-scope

int bit_and(int a, int b) { return a & b; }

// CHECK-LABEL: func.func @bit_and
// CHECK: arith.andi %arg0, %arg1 : i32

int bit_or(int a, int b) { return a | b; }

// CHECK-LABEL: func.func @bit_or
// CHECK: arith.ori %arg0, %arg1 : i32

int bit_xor(int a, int b) { return a ^ b; }

// CHECK-LABEL: func.func @bit_xor
// CHECK: arith.xori %arg0, %arg1 : i32

int bit_not(int value) { return ~value; }

// CHECK-LABEL: func.func @bit_not
// CHECK: arith.xori

int shift_left(int value, int count) { return value << count; }

// CHECK-LABEL: func.func @shift_left
// CHECK: arith.shli %arg0, %arg1 : i32

int shift_signed(int value, int count) { return value >> count; }

// CHECK-LABEL: func.func @shift_signed
// CHECK: arith.shrsi %arg0, %arg1 : i32
