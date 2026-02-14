// RUN: cmlirc %s -function=dot -mem2reg | FileCheck %s

int dot(int *a, int *b, int n) {
  int result = 0;
  for (int i = 0; i < n; i++) {
    result += a[i] * b[i];
  }
  return result;
}

// CHECK: func @dot(%arg0: memref<?xi32>, %arg1: memref<?xi32>, %arg2: i32) -> i32
// CHECK-DAG: %c0 = arith.constant 0 : i32
// CHECK-DAG: %c1 = arith.constant 1 : i32
// CHECK-DAG: %c0_i32 = arith.constant 0 : i32
// CHECK: %[[V0:.*]] = arith.index_cast %arg2 : i32 to index
// CHECK: %[[V1:.*]] = scf.for %arg3 = %c0 to %[[V0]] step %c1 iter_args(%arg4 = %c0_i32) -> (i32) {
// CHECK:   %[[V2:.*]] = memref.load %arg0[%arg3] : memref<?xi32>
// CHECK:   %[[V3:.*]] = memref.load %arg1[%arg3] : memref<?xi32>
// CHECK:   %[[V4:.*]] = arith.muli %[[V2]], %[[V3]] : i32
// CHECK:   %[[V5:.*]] = arith.addi %[[V4]], %arg4 : i32
// CHECK:   scf.yield %[[V5]] : i32
// CHECK: }
// CHECK: return %[[V1]] : i32
