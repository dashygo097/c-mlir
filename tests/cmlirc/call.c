// RUN: cmlirc %s | FileCheck %s

int callee(int a, int b) {
  return a + b;
}

int caller() {
  return callee(1, 2);
}

// CHECK: func.func @callee(%arg0: i32, %arg1: i32) -> i32
// CHECK: %[[V0:.*]] = arith.addi %arg0, %arg1 : i32
// CHECK: return %[[V0]] : i32

// CHECK: func.func @caller() -> i32
// CHECK-DAG: %[[c1_i32:.+]] = arith.constant 1 : i32
// CHECK-DAG: %[[c2_i32:.+]] = arith.constant 2 : i32
// CHECK: %[[V0:.*]] = call @callee(%c1_i32, %c2_i32) : (i32, i32) -> i32
// CHECK: return %[[V0]] : i32
