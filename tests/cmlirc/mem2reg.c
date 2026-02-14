// RUN: cmlirc %s -function=dot --mem2reg --symdce --cse --canonicalize --licm --sscp | FileCheck %s --check-prefix=CHECKDOT

int dot(int *a, int *b, int n) {
  int result = 0;
  for (int i = 0; i < n; i++) {
    result += a[i] * b[i];
  }
  return result;
}

// CHECKDOT: func @dot(%arg0: memref<?xi32>, %arg1: memref<?xi32>, %arg2: i32) -> i32
// CHECKDOT-DAG: %[[c0:.+]] = arith.constant 0 : index
// CHECKDOT-DAG: %[[c1:.+]] = arith.constant 1 : index
// CHECKDOT-DAG: %[[c0_i32:.+]] = arith.constant 0 : i32
// CHECKDOT: %[[V0:.*]] = arith.index_cast %arg2 : i32 to index
// CHECKDOT: %[[V1:.*]] = scf.for %arg3 = %c0 to %[[V0]] step %c1 iter_args(%arg4 = %c0_i32) -> (i32) {
// CHECKDOT:   %[[V2:.*]] = memref.load %arg0[%arg3] : memref<?xi32>
// CHECKDOT:   %[[V3:.*]] = memref.load %arg1[%arg3] : memref<?xi32>
// CHECKDOT:   %[[V4:.*]] = arith.muli %[[V2]], %[[V3]] : i32
// CHECKDOT:   %[[V5:.*]] = arith.addi %arg4, %[[V4]] : i32
// CHECKDOT:   scf.yield %[[V5]] : i32
// CHECKDOT: }
// CHECKDOT: return %[[V1]] : i32
