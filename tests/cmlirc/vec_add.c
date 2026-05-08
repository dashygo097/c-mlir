// RUN: split-file %s %t

// RUN: cmlirc %t/vec_add.c -function=vec_add | FileCheck %s
// RUN: cmlirc %t/vec_add.c -function=vec_add --raise-scf-to-affine | FileCheck %s --check-prefix=CHECKAFFINE
// RUN: cmlirc %t/vec_add_vectorize.c -function=vec_add_vec --raise-scf-to-affine | FileCheck %s --check-prefix=CHECKVEC

//--- vec_add.c
void vec_add(float *c, const float *a, const float *b, int n) {
  for (int i = 0; i < n; i++) {
    c[i] = a[i] + b[i];
  }
}

// CHECK: func.func @vec_add(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: memref<?xf32>, %arg3: i32)
// CHECK-DAG: %[[c0:.+]] = arith.constant 0 : index
// CHECK-DAG: %[[c1:.+]] = arith.constant 1 : index
// CHECK: %[[N:.+]] = arith.index_cast %arg3 : i32 to index
// CHECK: scf.for %[[I:.+]] = %[[c0]] to %[[N]] step %[[c1]] {
// CHECK:   %[[A:.+]] = memref.load %arg1[%[[I]]] : memref<?xf32>
// CHECK:   %[[B:.+]] = memref.load %arg2[%[[I]]] : memref<?xf32>
// CHECK:   %[[SUM:.+]] = arith.addf %[[A]], %[[B]] : f32
// CHECK:   memref.store %[[SUM]], %arg0[%[[I]]] : memref<?xf32>
// CHECK: }
// CHECK: return

// CHECKAFFINE: func.func @vec_add(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: memref<?xf32>, %arg3: i32)
// CHECKAFFINE: %[[N:.+]] = arith.index_cast %arg3 : i32 to index
// CHECKAFFINE: affine.for %[[I:.+]] = 0 to %[[N]] {
// CHECKAFFINE:   %[[A:.+]] = memref.load %arg1[%[[I]]] : memref<?xf32>
// CHECKAFFINE:   %[[B:.+]] = memref.load %arg2[%[[I]]] : memref<?xf32>
// CHECKAFFINE:   %[[SUM:.+]] = arith.addf %[[A]], %[[B]] : f32
// CHECKAFFINE:   memref.store %[[SUM]], %arg0[%[[I]]] : memref<?xf32>
// CHECKAFFINE: }
// CHECKAFFINE: return

//--- vec_add_vectorize.c
void vec_add_vec(float *c, const float *a, const float *b, int n) {
#pragma cmlir loop vectorize(enable) vectorize_width(2)
  for (int i = 0; i < n; i++) {
    c[i] = a[i] + b[i];
  }
}

// CHECKVEC-DAG: #[[MAP0:.+]] = affine_map<()[s0] -> (s0 ceildiv 2)>
// CHECKVEC-DAG: #[[MAP1:.+]] = affine_map<(d0) -> (d0 * 2)>
// CHECKVEC: func.func @vec_add_vec(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: memref<?xf32>, %arg3: i32)
// CHECKVEC: %[[N:.+]] = arith.index_cast %arg3 : i32 to index
// CHECKVEC: affine.for %[[I:.+]] = 0 to #[[MAP0]]()[%[[N]]] {
// CHECKVEC:   %[[IDX:.+]] = affine.apply #[[MAP1]](%[[I]])
// CHECKVEC:   %[[A:.+]] = vector.load %arg1[%[[IDX]]] : memref<?xf32>, vector<2xf32>
// CHECKVEC:   %[[B:.+]] = vector.load %arg2[%[[IDX]]] : memref<?xf32>, vector<2xf32>
// CHECKVEC:   %[[SUM:.+]] = arith.addf %[[A]], %[[B]] : vector<2xf32>
// CHECKVEC:   vector.store %[[SUM]], %arg0[%[[IDX]]] : memref<?xf32>, vector<2xf32>
// CHECKVEC: }
// CHECKVEC: return
