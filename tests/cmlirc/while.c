// RUN: cmlirc %s -function=_while --canonicalize --cse --symdce --mem2reg --const-prop | FileCheck %s

int _while() {
  int i = 0;

  while (i < 5) {
    i++;
  }

  return i;
}

// CHECK: func.func @_while() -> i32
// CHECK-DAG: %[[c0_i32:.*]] = arith.constant 0 : i32
// CHECK-DAG: %[[c1_i32:.*]] = arith.constant 1 : i32
// CHECK-DAG: %[[c5_i32:.*]] = arith.constant 5 : i32
// CHECK: %[[V0:.*]] = scf.while (%arg0 = %[[c0_i32]]) : (i32) -> i32 {
// CHECK:   %[[V1:.*]] = arith.cmpi slt, %arg0, %[[c5_i32]] : i32
// CHECK:   scf.condition(%[[V1]]) %arg0 : i32
// CHECK: } do {
// CHECK: ^bb0(%arg0: i32):
// CHECK:   %[[V1:.*]] = arith.addi %arg0, %[[c1_i32]] : i32
// CHECK:   scf.yield %[[V1]] : i32
// CHECK: }
// CHECK: return %0 : i32
