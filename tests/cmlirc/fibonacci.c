// RUN: cmlirc %s -function=fibonacci | FileCheck %s

int fibonacci(int n) {
  if (n <= 1) {
    return n;
  }

  return fibonacci(n - 1) + fibonacci(n - 2);
}

// CHECK: func.func @fibonacci(%arg0: i32) -> i32
// CHECK-DAG: %[[c1_i32:.*]] = arith.constant 1 : i32
// CHECK-DAG: %[[c2_i32:.*]] = arith.constant 2 : i32
// CHECK: %[[V0:.*]] = arith.cmpi sle, %arg0, %[[c1_i32]] : i32
// CHECK: cf.cond_br %[[V0]], ^bb{{[0-9]+}}, ^bb{{[0-9]+}}
// CHECK: %[[V1:.*]] = arith.subi %arg0, %[[c1_i32]] : i32
// CHECK: %[[V2:.*]] = call @fibonacci(%[[V1]]) : (i32) -> i32
// CHECK: %[[V3:.*]] = arith.subi %arg0, %[[c2_i32]] : i32
// CHECK: %[[V4:.*]] = call @fibonacci(%[[V3]]) : (i32) -> i32
// CHECK: %[[V5:.*]] = arith.addi %[[V2]], %[[V4]] : i32
// CHECK: return %[[V5]] : i32
// CHECK: return %arg0 : i32
