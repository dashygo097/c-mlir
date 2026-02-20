// RUN: cmlirc %s -function=mlp --mem2reg --licm --canonicalize --sscp --symdce --cse --const-prop | FileCheck %s

void mlp(float *output, float *input) {
  float weights[2][2] = {{0.1, 0.2}, {0.3, 0.4}};
  float bias[2] = {0.1, 0.2};

  for (int i = 0; i < 2; i++) {
    output[i] = bias[i];
    for (int j = 0; j < 2; j++) {
      output[i] += weights[i][j] * input[j];
    }
  }
}

// CHECK: func @mlp(%arg0: memref<?xf32>, %arg1: memref<?xf32>)
// CHECK-DAG: %[[c0:.+]] = arith.constant 0 : index
// CHECK-DAG: %[[c1:.+]] = arith.constant 1 : index
// CHECK-DAG: %[[c2:.+]] = arith.constant 2 : index
// CHECK-DAG: %[[cst:.+]] = arith.constant 1.000000e-01 : f32
// CHECK-DAG: %[[cst_0:.+]] = arith.constant 2.000000e-01 : f32
// CHECK-DAG: %[[cst_1:.+]] = arith.constant 3.000000e-01 : f32
// CHECK-DAG: %[[cst_2:.+]] = arith.constant 4.000000e-01 : f32
// CHECK: %[[alloca:.+]] = memref.alloca() : memref<2x2xf32>
// CHECK: affine.store %[[cst]], %[[alloca]][0, 0] : memref<2x2xf32>
// CHECK: affine.store %[[cst_0]], %[[alloca]][0, 1] : memref<2x2xf32>
// CHECK: affine.store %[[cst_1]], %[[alloca]][1, 0] : memref<2x2xf32>
// CHECK: affine.store %[[cst_2]], %[[alloca]][1, 1] : memref<2x2xf32>
// CHECK: %[[alloca_3:.+]] = memref.alloca() : memref<2xf32>
// CHECK: affine.store %[[cst]], %[[alloca_3]][0] : memref<2xf32>
// CHECK: affine.store %[[cst_0]], %[[alloca_3]][1] : memref<2xf32>
// CHECK: scf.for %[[arg2:.+]] = %[[c0]] to %[[c2]] step %[[c1]] {
// CHECK:   %[[V0:.+]] = memref.load %[[alloca_3]][%[[arg2]]] : memref<2xf32>
// CHECK:   memref.store %[[V0]], %arg0[%[[arg2]]] : memref<?xf32>
// CHECK:   scf.for %[[arg3:.+]] = %[[c0]] to %[[c2]] step %[[c1]] {
// CHECK:     %[[V1:.+]] = memref.load %[[alloca]][%[[arg2]], %[[arg3]]] : memref<2x2xf32>
// CHECK:     %[[V2:.+]] = memref.load %arg1[%[[arg3]]] : memref<?xf32>
// CHECK:     %[[V3:.+]] = arith.mulf %[[V1]], %[[V2]] : f32
// CHECK:     %[[V4:.+]] = memref.load %arg0[%[[arg2]]] : memref<?xf32>
// CHECK:     %[[V5:.+]] = arith.addf %[[V4]], %[[V3]] : f32
// CHECK:     memref.store %[[V5]], %arg0[%[[arg2]]] : memref<?xf32>
// CHECK:   }
// CHECK: }
