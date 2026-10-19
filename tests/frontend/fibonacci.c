// RUN: clang-to-mlir %s -function=fibonacci | FileCheck %s --implicit-check-not=cf.cond_br --implicit-check-not=cf.br

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
// CHECK: %[[RESULT:.*]] = scf.if %[[V0]] -> (i32) {
// CHECK: scf.yield %arg0 : i32
// CHECK: } else {
// CHECK: %[[V1:.*]] = arith.subi %arg0, %[[c1_i32]] : i32
// CHECK: %[[V2:.*]] = func.call @fibonacci(%[[V1]]) : (i32) -> i32
// CHECK: %[[V3:.*]] = arith.subi %arg0, %[[c2_i32]] : i32
// CHECK: %[[V4:.*]] = func.call @fibonacci(%[[V3]]) : (i32) -> i32
// CHECK: %[[V5:.*]] = arith.addi %[[V2]], %[[V4]] : i32
// CHECK: scf.yield %[[V5]] : i32
// CHECK: }
// CHECK: return %[[RESULT]] : i32
