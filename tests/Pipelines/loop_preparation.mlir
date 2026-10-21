// RUN: cmlir-opt %s --default-pipeline --verify-each | FileCheck %s

// Fold the full-tile size using the loop bounds and step.
// CHECK-LABEL: func.func @canonicalize_tile_size(
// CHECK-DAG: %[[TWO:.*]] = arith.constant 2 : index
// CHECK: scf.for %[[IV:.*]] =
// CHECK-NOT: affine.min
// CHECK: memref.store %[[TWO]], %arg0[%[[IV]]] : memref<?xindex>
// CHECK: return
func.func @canonicalize_tile_size(%out: memref<?xindex>) {
  %c0 = arith.constant 0 : index
  %c2 = arith.constant 2 : index
  %c8 = arith.constant 8 : index
  scf.for %i = %c0 to %c8 step %c2 {
    %size = affine.min affine_map<(d0) -> (2, 8 - d0)> (%i)
    memref.store %size, %out[%i] : memref<?xindex>
  }
  return
}

// Fold i * 2 + 3 into the bounds and step before transforming the loop.
// CHECK-LABEL: func.func @fold_index_arithmetic(
// CHECK-DAG: %[[LOWER:.*]] = arith.constant 3 : index
// CHECK-DAG: %[[UPPER:.*]] = arith.constant 11 : index
// CHECK-DAG: %[[STEP:.*]] = arith.constant 2 : index
// CHECK: scf.for %[[IV:.*]] = %[[LOWER]] to %[[UPPER]] step %[[STEP]] {
// CHECK-NEXT: memref.store {{.*}}, %arg0[%[[IV]]] : memref<?xindex>
// CHECK-NEXT: }
// CHECK: return
func.func @fold_index_arithmetic(%out: memref<?xindex>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %c4 = arith.constant 4 : index
  scf.for %i = %c0 to %c4 step %c1 {
    %scaled = arith.muli %i, %c2 : index
    %offset = arith.addi %scaled, %c3 : index
    memref.store %c1, %out[%offset] : memref<?xindex>
  }
  return
}

// A negative multiplier must not become a negative scf.for step.
// CHECK-LABEL: func.func @keep_negative_multiplier(
// CHECK-DAG: %[[NEGATIVE:.*]] = arith.constant -2 : index
// CHECK: scf.for %[[IV:.*]] =
// CHECK-NEXT: %[[VALUE:.*]] = arith.muli %[[IV]], %[[NEGATIVE]] : index
// CHECK-NEXT: memref.store %[[VALUE]], %arg0[] : memref<index>
// CHECK: return
func.func @keep_negative_multiplier(%out: memref<index>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %negative = arith.constant -2 : index
  scf.for %i = %c0 to %c4 step %c1 {
    %value = arith.muli %i, %negative : index
    memref.store %value, %out[] : memref<index>
  }
  return
}

// Keep the original induction variable when it also indexes the store.
// CHECK-LABEL: func.func @keep_multiple_users(
// CHECK-DAG: %[[TWO:.*]] = arith.constant 2 : index
// CHECK: scf.for %[[IV:.*]] =
// CHECK-NEXT: %[[VALUE:.*]] = arith.muli %[[IV]], %[[TWO]] : index
// CHECK-NEXT: memref.store %[[VALUE]], %arg0[%[[IV]]] : memref<?xindex>
// CHECK: return
func.func @keep_multiple_users(%out: memref<?xindex>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c4 = arith.constant 4 : index
  scf.for %i = %c0 to %c4 step %c1 {
    %value = arith.muli %i, %c2 : index
    memref.store %value, %out[%i] : memref<?xindex>
  }
  return
}
