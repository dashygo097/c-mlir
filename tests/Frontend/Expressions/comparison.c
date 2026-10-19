// RUN: clang-to-mlir --disable-opt %s -- | FileCheck %s --enable-var-scope

int signed_less(int a, int b) { return a < b; }

// CHECK-LABEL: func.func @signed_less
// CHECK: %[[C:.*]] = arith.cmpi slt, %arg0, %arg1 : i32
// CHECK: %[[R:.*]] = arith.extui %[[C]] : i1 to i32
// CHECK: return %[[R]] : i32

int float_less(float a, float b) { return a < b; }

// CHECK-LABEL: func.func @float_less
// CHECK: arith.cmpf olt

int float_not_equal(float a, float b) { return a != b; }

// CHECK-LABEL: func.func @float_not_equal
// CHECK: arith.cmpf une
