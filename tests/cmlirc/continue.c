// RUN: cmlirc %s -function=reduce_sum_positive | FileCheck %s

float reduce_sum_positive(float *a, int n) {
  float result = 0.0f;
  for (int i = 0; i < n; i++) {
    if (a[i] == 0.0f) {
      continue;
    }
    result += a[i];
  }
  return result;
}

// CHECK: func.func @reduce_sum_positive(%arg0: memref<?xf32>, %arg1: i32) -> f32
// CHECK-DAG: %[[cst:.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG: %[[c0_i32:.+]] = arith.constant 0 : i32
// CHECK-DAG: %[[c1_i32:.+]] = arith.constant 1 : i32
// CHECK: %[[V0:.*]]:2 = scf.while (%arg2 = %[[cst]], %arg3 = %[[c0_i32]]) : (f32, i32) -> (f32, i32) {
// CHECK:   %[[V1:.*]] = arith.cmpi slt, %arg3, %arg1 : i32
// CHECK:   scf.condition(%[[V1]]) %arg2, %arg3 : f32, i32
// CHECK: } do {
// CHECK: ^bb0(%arg2: f32, %arg3: i32):
// CHECK:   %[[V2:.*]] = arith.index_cast %arg3 : i32 to index
// CHECK:   %[[V3:.*]] = memref.load %arg0[%[[V2]]] : memref<?xf32>
// CHECK:   %[[V4:.*]] = arith.cmpf oeq, %[[V3]], %[[cst]] : f32
// CHECK:   %[[V5:.*]] = scf.if %[[V4]] -> (f32) {
// CHECK:     scf.yield %arg2 : f32
// CHECK:   } else {
// CHECK:     %[[V7:.*]] = memref.load %arg0[%[[V2]]] : memref<?xf32>
// CHECK:     %[[V8:.*]] = arith.addf %arg2, %[[V7]] : f32
// CHECK:     scf.yield %[[V8]] : f32
// CHECK:   }
// CHECK:   %[[V6:.*]] = arith.addi %arg3, %[[c1_i32]] : i32
// CHECK:   scf.yield %[[V5]], %[[V6]] : f32, i32
// CHECK: }
// CHECK: return %[[V0]]#0 : f32
