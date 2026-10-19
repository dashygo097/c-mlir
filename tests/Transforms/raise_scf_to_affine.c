// RUN: clang-to-mlir %s --raise-scf-to-affine -- | FileCheck %s

void vec_add(float *c, const float *a, const float *b, int n) {
  for (int i = 0; i < n; i++) {
    c[i] = a[i] + b[i];
  }
}

// CHECK: func.func @vec_add(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: memref<?xf32>, %arg3: i32)
// CHECK: %[[N:.*]] = arith.index_cast %arg3 : i32 to index
// CHECK: affine.for %[[I:.*]] = 0 to %[[N]] {
// CHECK:   %[[A:.*]] = memref.load %arg1[%[[I]]] : memref<?xf32>
// CHECK:   %[[B:.*]] = memref.load %arg2[%[[I]]] : memref<?xf32>
// CHECK:   %[[SUM:.*]] = arith.addf %[[A]], %[[B]] : f32
// CHECK:   memref.store %[[SUM]], %arg0[%[[I]]] : memref<?xf32>
// CHECK: }
// CHECK: return
