// RUN: clang-to-mlir %s -- | FileCheck %s --check-prefix=CHECKVEC
// RUN: clang-to-mlir --disable-opt %s -- | cmlir-opt --pass-pipeline='builtin.module(default-pipeline)' | FileCheck %s --check-prefix=CHECKVEC
// RUN: clang-to-mlir %s --raise-scf-to-affine -- | FileCheck %s --check-prefix=CHECKVECAFFINE
// RUN: clang-to-mlir --disable-opt %s -- | cmlir-opt --pass-pipeline='builtin.module(default-pipeline{raise-scf-to-affine=true})' | FileCheck %s --check-prefix=CHECKVECAFFINE

void vec_add_vec(float *__restrict c, const float *a, const float *b, int n) {
#pragma cmlir loop vectorize(enable) vectorize_width(4)
  for (int i = 0; i < n; i++) {
    c[i] = a[i] + b[i];
  }
}

// CHECKVEC: func.func @vec_add_vec(%arg0: memref<?xf32, strided<[1], offset: ?>> {cmlir.restrict}, %arg1: memref<?xf32, strided<[1], offset: ?>>, %arg2: memref<?xf32, strided<[1], offset: ?>>, %arg3: i32)
// CHECKVEC-DAG: %[[c4:.*]] = arith.constant 4 : index
// CHECKVEC-DAG: %[[c1:.*]] = arith.constant 1 : index
// CHECKVEC-DAG: %[[c0:.*]] = arith.constant 0 : index
// CHECKVEC: %[[N:.*]] = arith.index_cast %arg3 : i32 to index
// CHECKVEC: scf.for %[[TILE:.*]] = %[[c0]] to %[[N]] step %[[c4]] {
// CHECKVEC: scf.if
// CHECKVEC:   %[[A:.*]] = vector.transfer_read %arg1[{{.*}}], {{.*}} : memref<?xf32, strided<[1], offset: ?>>, vector<4xf32>
// CHECKVEC:   %[[B:.*]] = vector.transfer_read %arg2[{{.*}}], {{.*}} : memref<?xf32, strided<[1], offset: ?>>, vector<4xf32>
// CHECKVEC:   %[[SUM:.*]] = arith.addf %[[A]], %[[B]] : vector<4xf32>
// CHECKVEC:   vector.transfer_write %[[SUM]], %arg0[{{.*}}] : vector<4xf32>, memref<?xf32, strided<[1], offset: ?>>
// CHECKVEC: } else {
// CHECKVEC: scf.for %[[RI:.*]] = %[[TILE]] to {{.*}} step %[[c1]] {
// CHECKVEC:   %[[RA:.*]] = memref.load %arg1[%[[RI]]] : memref<?xf32, strided<[1], offset: ?>>
// CHECKVEC:   %[[RB:.*]] = memref.load %arg2[%[[RI]]] : memref<?xf32, strided<[1], offset: ?>>
// CHECKVEC:   %[[RSUM:.*]] = arith.addf %[[RA]], %[[RB]] : f32
// CHECKVEC:   memref.store %[[RSUM]], %arg0[%[[RI]]] : memref<?xf32, strided<[1], offset: ?>>
// CHECKVEC: }
// CHECKVEC: return

// CHECKVECAFFINE-LABEL: func.func @vec_add_vec(
// CHECKVECAFFINE: affine.for
// CHECKVECAFFINE: vector.transfer_read {{.*}} : memref<?xf32, strided<[1], offset: ?>>, vector<4xf32>
// CHECKVECAFFINE: vector.transfer_read {{.*}} : memref<?xf32, strided<[1], offset: ?>>, vector<4xf32>
// CHECKVECAFFINE: arith.addf {{.*}} : vector<4xf32>
// CHECKVECAFFINE: vector.transfer_write
// CHECKVECAFFINE: memref.load
// CHECKVECAFFINE: memref.store
// CHECKVECAFFINE: return
