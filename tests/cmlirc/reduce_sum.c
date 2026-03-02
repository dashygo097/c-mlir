// RUN: split-file %s %t

// RUN: cmlirc %t/reduce_sum.c -function=reduce_sum64 | FileCheck %s
// RUN: cmlirc %t/reduce_sum_loop_unroll.c -function=reduce_sum64 | FileCheck %s --check-prefix=CHECKUNROLL

//--- reduce_sum.c
float reduce_sum64(float a[64]) {
  float result = 0.0f;
  for (int i = 0; i < 64; i++) {
    result += a[i];
  }
  return result;
}

// CHECK: func @reduce_sum64(%arg0: memref<?xf32>) -> f32
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

//--- reduce_sum_loop_unroll.c
float reduce_sum64(float a[64]) {
  float result = 0.0f;
#pragma cmlir loop unroll(2)
  for (int i = 0; i < 64; i++) {
    result += a[i];
  }
  return result;
}

// CHECKUNROLL: func @reduce_sum64(%arg0: memref<?xf32>) -> f32
// CHECKUNROLL-DAG: %[[c0:.+]] = arith.constant 0 : index
// CHECKUNROLL-DAG: %[[c1:.+]] = arith.constant 1 : index
// CHECKUNROLL-DAG: %[[c2:.+]] = arith.constant 2 : index
// CHECKUNROLL-DAG: %[[c64:.+]] = arith.constant 64 : index
// CHECKUNROLL-DAG: %[[cst:.+]] = arith.constant 0.000000e+00 : f32
// CHECKUNROLL: %[[V0:.*]] = scf.for %arg1 = %[[c0]] to %[[c64]] step %[[c2]] iter_args(%arg2 = %[[cst]]) -> (f32) {
// CHECKUNROLL:   %[[V1:.*]] = memref.load %arg0[%arg1] : memref<?xf32>
// CHECKUNROLL:   %[[V2:.*]] = arith.addf %arg2, %[[V1]] : f32
// CHECKUNROLL:   %[[V3:.*]] = arith.addi %arg1, %[[c1]] : index
// CHECKUNROLL:   %[[V4:.*]] = memref.load %arg0[%[[V3]]] : memref<?xf32>
// CHECKUNROLL:   %[[V5:.*]] = arith.addf %[[V2]], %[[V4]] : f32
// CHECKUNROLL:   scf.yield %[[V5]] : f32
// CHECKUNROLL: }
// CHECKUNROLL: return %[[V0]] : f32
