// RUN: clang-to-mlir %s -function=reduce_sum_nonzero | FileCheck %s --implicit-check-not=cf.cond_br --implicit-check-not=cf.br

float reduce_sum_nonzero(float *a, int n) {
  float result = 0.0f;
  for (int i = 0; i < n; i++) {
    if (a[i] == 0.0f) {
      continue;
    }
    result += a[i];
  }
  return result;
}

// CHECK: func.func @reduce_sum_nonzero(%arg0: memref<?xf32>, %arg1: i32) -> f32
// CHECK-DAG: %[[cst:.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG: %[[c0_i32:.+]] = arith.constant 0 : i32
// CHECK-DAG: %[[c1_i32:.+]] = arith.constant 1 : i32
// CHECK: %[[LOOP:.*]]:2 = scf.while (%[[SUM:.*]] = %[[cst]], %[[IV:.*]] = %[[c0_i32]]) : (f32, i32) -> (f32, i32) {
// CHECK: %[[COND:.*]] = arith.cmpi slt, %[[IV]], %arg1 : i32
// CHECK: scf.condition(%[[COND]]) %[[SUM]], %[[IV]] : f32, i32
// CHECK: } do {
// CHECK: ^bb0(%[[BODY_SUM:.*]]: f32, %[[BODY_IV:.*]]: i32):
// CHECK: %[[INDEX:.*]] = arith.index_cast %[[BODY_IV]] : i32 to index
// CHECK: %[[ELEMENT:.*]] = memref.load %arg0[%[[INDEX]]] : memref<?xf32>
// CHECK: %[[SKIP:.*]] = arith.cmpf oeq, %[[ELEMENT]], %[[cst]] : f32
// CHECK: %[[NEXT_SUM:.*]] = scf.if %[[SKIP]] -> (f32) {
// CHECK: scf.yield %[[BODY_SUM]] : f32
// CHECK: } else {
// CHECK: %[[VALUE:.*]] = memref.load %arg0[%[[INDEX]]] : memref<?xf32>
// CHECK: %[[ADDED:.*]] = arith.addf %[[BODY_SUM]], %[[VALUE]] : f32
// CHECK: scf.yield %[[ADDED]] : f32
// CHECK: }
// CHECK: %[[NEXT_IV:.*]] = arith.addi %[[BODY_IV]], %[[c1_i32]] : i32
// CHECK: scf.yield %[[NEXT_SUM]], %[[NEXT_IV]] : f32, i32
// CHECK: }
// CHECK: return %[[LOOP]]#0 : f32
