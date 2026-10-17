// RUN: split-file %s %t
//
// RUN: clang-to-mlir %t/vec_add.c -function=vec_add | FileCheck %s --check-prefix=CHECK
// RUN: clang-to-mlir %t/vec_add.c -function=vec_add --raise-scf-to-affine | FileCheck %s --check-prefix=CHECKAFFINE
// RUN: clang-to-mlir %t/vec_add_vectorize.c -function=vec_add_vec | FileCheck %s --check-prefix=CHECKVEC
// RUN: clang-to-mlir %t/vec_add_vectorize.c -function=vec_add_vec --raise-scf-to-affine | FileCheck %s --check-prefix=CHECKVECAFFINE

//--- vec_add.c
void vec_add(float *c, const float *a, const float *b, int n) {
  for (int i = 0; i < n; i++) {
    c[i] = a[i] + b[i];
  }
}

// CHECK: func.func @vec_add(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: memref<?xf32>, %arg3: i32)
// CHECK-DAG: %[[c0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[c1:.*]] = arith.constant 1 : index
// CHECK: %[[N:.*]] = arith.index_cast %arg3 : i32 to index
// CHECK: scf.for %[[I:.*]] = %[[c0]] to %[[N]] step %[[c1]] {
// CHECK:   %[[A:.*]] = memref.load %arg1[%[[I]]] : memref<?xf32>
// CHECK:   %[[B:.*]] = memref.load %arg2[%[[I]]] : memref<?xf32>
// CHECK:   %[[SUM:.*]] = arith.addf %[[A]], %[[B]] : f32
// CHECK:   memref.store %[[SUM]], %arg0[%[[I]]] : memref<?xf32>
// CHECK: }
// CHECK: return

// CHECKAFFINE: func.func @vec_add(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: memref<?xf32>, %arg3: i32)
// CHECKAFFINE: %[[N:.*]] = arith.index_cast %arg3 : i32 to index
// CHECKAFFINE: affine.for %[[I:.*]] = 0 to %[[N]] {
// CHECKAFFINE:   %[[A:.*]] = memref.load %arg1[%[[I]]] : memref<?xf32>
// CHECKAFFINE:   %[[B:.*]] = memref.load %arg2[%[[I]]] : memref<?xf32>
// CHECKAFFINE:   %[[SUM:.*]] = arith.addf %[[A]], %[[B]] : f32
// CHECKAFFINE:   memref.store %[[SUM]], %arg0[%[[I]]] : memref<?xf32>
// CHECKAFFINE: }
// CHECKAFFINE: return

//--- vec_add_vectorize.c
void vec_add_vec(float *__restrict c, const float *a, const float *b, int n) {
#pragma cmlir loop vectorize(enable) vectorize_width(4)
  for (int i = 0; i < n; i++) {
    c[i] = a[i] + b[i];
  }
}

// CHECKVEC: func.func @vec_add_vec(%arg0: memref<?xf32> {cmlir.restrict}, %arg1: memref<?xf32>, %arg2: memref<?xf32>, %arg3: i32)
// CHECKVEC-DAG: %[[c4:.*]] = arith.constant 4 : index
// CHECKVEC-DAG: %[[c1:.*]] = arith.constant 1 : index
// CHECKVEC-DAG: %[[c0:.*]] = arith.constant 0 : index
// CHECKVEC: %[[N:.*]] = arith.index_cast %arg3 : i32 to index
// CHECKVEC: scf.for %[[TILE:.*]] = %[[c0]] to %[[N]] step %[[c4]] {
// CHECKVEC: scf.if
// CHECKVEC:   %[[A:.*]] = vector.transfer_read %arg1[{{.*}}], {{.*}} : memref<?xf32>, vector<4xf32>
// CHECKVEC:   %[[B:.*]] = vector.transfer_read %arg2[{{.*}}], {{.*}} : memref<?xf32>, vector<4xf32>
// CHECKVEC:   %[[SUM:.*]] = arith.addf %[[A]], %[[B]] : vector<4xf32>
// CHECKVEC:   vector.transfer_write %[[SUM]], %arg0[{{.*}}] : vector<4xf32>, memref<?xf32>
// CHECKVEC: } else {
// CHECKVEC: scf.for %[[RI:.*]] = %[[TILE]] to {{.*}} step %[[c1]] {
// CHECKVEC:   %[[RA:.*]] = memref.load %arg1[%[[RI]]] : memref<?xf32>
// CHECKVEC:   %[[RB:.*]] = memref.load %arg2[%[[RI]]] : memref<?xf32>
// CHECKVEC:   %[[RSUM:.*]] = arith.addf %[[RA]], %[[RB]] : f32
// CHECKVEC:   memref.store %[[RSUM]], %arg0[%[[RI]]] : memref<?xf32>
// CHECKVEC: }
// CHECKVEC: return

// CHECKVECAFFINE-LABEL: func.func @vec_add_vec(
// CHECKVECAFFINE: affine.for
// CHECKVECAFFINE: vector.transfer_read {{.*}} : memref<?xf32>, vector<4xf32>
// CHECKVECAFFINE: vector.transfer_read {{.*}} : memref<?xf32>, vector<4xf32>
// CHECKVECAFFINE: arith.addf {{.*}} : vector<4xf32>
// CHECKVECAFFINE: vector.transfer_write
// CHECKVECAFFINE: memref.load
// CHECKVECAFFINE: memref.store
// CHECKVECAFFINE: return
