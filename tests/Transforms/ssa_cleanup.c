// RUN: clang-to-mlir %s --function=dot -- | FileCheck %s --check-prefix=DOT
// RUN: clang-to-mlir %s --function=do_while -- | FileCheck %s --check-prefix=DO
// RUN: clang-to-mlir %s --function=_while -- | FileCheck %s --check-prefix=WHILE
// RUN: clang-to-mlir %s --function=reduce_sum64 -- | FileCheck %s
// RUN: clang-to-mlir %s --function=vec_add -- | FileCheck %s --check-prefix=VEC

float reduce_sum64(float a[64]) {
  float result = 0.0f;
  for (int i = 0; i < 64; i++) {
    result += a[i];
  }
  return result;
}

// CHECK: func.func @reduce_sum64(%arg0: memref<?xf32>) -> f32
// CHECK-DAG: %[[c0:.+]] = arith.constant 0 : index
// CHECK-DAG: %[[c1:.+]] = arith.constant 1 : index
// CHECK-DAG: %[[c64:.+]] = arith.constant 64 : index
// CHECK-DAG: %[[cst:.+]] = arith.constant 0.000000e+00 : f32
// CHECK: %[[V0:.*]] = scf.for %arg1 = %[[c0]] to %[[c64]] step %c1 iter_args(%arg2 = %[[cst]]) -> (f32) {
// CHECK:   %[[V1:.*]] = memref.load %arg0[%arg1] : memref<?xf32>
// CHECK:   %[[V2:.*]] = arith.addf %arg2, %[[V1]] : f32
// CHECK:   scf.yield %[[V2]] : f32
// CHECK: }
// CHECK: return %[[V0]] : f32

void vec_add(float *c, const float *a, const float *b, int n) {
  for (int i = 0; i < n; i++) {
    c[i] = a[i] + b[i];
  }
}

// VEC: func.func @vec_add(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: memref<?xf32>, %arg3: i32)
// VEC-DAG: %[[c0:.*]] = arith.constant 0 : index
// VEC-DAG: %[[c1:.*]] = arith.constant 1 : index
// VEC: %[[N:.*]] = arith.index_cast %arg3 : i32 to index
// VEC: scf.for %[[I:.*]] = %[[c0]] to %[[N]] step %[[c1]] {
// VEC:   %[[A:.*]] = memref.load %arg1[%[[I]]] : memref<?xf32>
// VEC:   %[[B:.*]] = memref.load %arg2[%[[I]]] : memref<?xf32>
// VEC:   %[[SUM:.*]] = arith.addf %[[A]], %[[B]] : f32
// VEC:   memref.store %[[SUM]], %arg0[%[[I]]] : memref<?xf32>
// VEC: }
// VEC: return

int _while() {
  int i = 0;
  while (i < 5) {
    i++;
  }
  return i;
}

// WHILE: func.func @_while() -> i32
// WHILE-DAG: %[[c0_i32:.*]] = arith.constant 0 : i32
// WHILE-DAG: %[[c1_i32:.*]] = arith.constant 1 : i32
// WHILE-DAG: %[[c5_i32:.*]] = arith.constant 5 : i32
// WHILE: %[[V0:.*]] = scf.while (%arg0 = %[[c0_i32]]) : (i32) -> i32 {
// WHILE:   %[[V1:.*]] = arith.cmpi slt, %arg0, %[[c5_i32]] : i32
// WHILE:   scf.condition(%[[V1]]) %arg0 : i32
// WHILE: } do {
// WHILE: ^bb0(%arg0: i32):
// WHILE:   %[[V1:.*]] = arith.addi %arg0, %[[c1_i32]] : i32
// WHILE:   scf.yield %[[V1]] : i32
// WHILE: }
// WHILE: return %0 : i32

int do_while() {
  int i = 0;
  do {
    i++;
  } while (i < 5);
  return i;
}

// DO: func.func @do_while() -> i32
// DO-DAG: %[[c0_i32:.*]] = arith.constant 0 : i32
// DO-DAG: %[[c1_i32:.*]] = arith.constant 1 : i32
// DO-DAG: %[[c5_i32:.*]] = arith.constant 5 : i32
// DO: %[[V0:.*]] = scf.while (%arg0 = %[[c0_i32]]) : (i32) -> i32 {
// DO:   %[[V1:.*]] = arith.addi %arg0, %[[c1_i32]] : i32
// DO:   %[[V2:.*]] = arith.cmpi slt, %[[V1]], %[[c5_i32]] : i32
// DO:   scf.condition(%[[V2]]) %[[V1]] : i32
// DO: } do {
// DO: ^bb0(%arg0: i32):
// DO:   scf.yield %arg0 : i32
// DO: }
// DO: return %0 : i32

int dot(const int *a, const int *b, int n) {
  int result = 0;
  for (int i = 0; i < n; i++) {
    result += a[i] * b[i];
  }
  return result;
}

// DOT: func.func @dot(%arg0: memref<?xi32>, %arg1: memref<?xi32>, %arg2: i32) -> i32
// DOT-DAG: %[[c0:.+]] = arith.constant 0 : index
// DOT-DAG: %[[c1:.+]] = arith.constant 1 : index
// DOT-DAG: %[[c0_i32:.+]] = arith.constant 0 : i32
// DOT: %[[V0:.*]] = arith.index_cast %arg2 : i32 to index
// DOT: %[[V1:.*]] = scf.for %arg3 = %c0 to %[[V0]] step %c1 iter_args(%arg4 = %c0_i32) -> (i32) {
// DOT:   %[[V2:.*]] = memref.load %arg0[%arg3] : memref<?xi32>
// DOT:   %[[V3:.*]] = memref.load %arg1[%arg3] : memref<?xi32>
// DOT:   %[[V4:.*]] = arith.muli %[[V2]], %[[V3]] : i32
// DOT:   %[[V5:.*]] = arith.addi %arg4, %[[V4]] : i32
// DOT:   scf.yield %[[V5]] : i32
// DOT: }
// DOT: return %[[V1]] : i32
