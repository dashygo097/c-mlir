// RUN: clang-to-mlir --disable-opt %s -- | FileCheck %s --enable-var-scope

int add(int a, int b) { return a + b; }

// CHECK-LABEL: func.func @add
// CHECK: %[[R:.*]] = arith.addi %arg0, %arg1 : i32
// CHECK: return %[[R]] : i32

int subtract(int a, int b) { return a - b; }

// CHECK-LABEL: func.func @subtract
// CHECK: arith.subi %arg0, %arg1 : i32

int multiply(int a, int b) { return a * b; }

// CHECK-LABEL: func.func @multiply
// CHECK: arith.muli %arg0, %arg1 : i32

int divide(int a, int b) { return a / b; }

// CHECK-LABEL: func.func @divide
// CHECK: arith.divsi %arg0, %arg1 : i32

int modulo(int a, int b) { return a % b; }

// CHECK-LABEL: func.func @modulo
// CHECK: arith.remsi %arg0, %arg1 : i32

float floating(float a, float b) { return (a + b) * (a - b) / b; }

// CHECK-LABEL: func.func @floating
// CHECK: arith.addf
// CHECK: arith.subf
// CHECK: arith.mulf
// CHECK: arith.divf

int negate(int value) { return -value; }

// CHECK-LABEL: func.func @negate
// CHECK: arith.subi
