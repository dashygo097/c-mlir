// RUN: cmlir-opt %s --loop-unroll | FileCheck %s

// CHECK-LABEL: func.func @partial
// CHECK-DAG: %[[TWO:.*]] = arith.constant 2 : index
// CHECK: scf.for {{.*}} step %[[TWO]]
// CHECK: memref.store
// CHECK: memref.store
// CHECK: return
func.func @partial(%a: memref<16xindex>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  scf.for %i = %c0 to %c16 step %c1 {
    memref.store %i, %a[%i] : memref<16xindex>
  } {unroll_count = 2 : i64}
  return
}

// CHECK-LABEL: func.func @full
// CHECK-NOT: scf.for
// CHECK: memref.store
// CHECK: memref.store
// CHECK-NOT: scf.for
// CHECK: return
func.func @full(%a: memref<2xindex>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  scf.for %i = %c0 to %c2 step %c1 {
    memref.store %i, %a[%i] : memref<2xindex>
  } {unroll}
  return
}

// Disabling unrolling takes precedence over a count hint.
// CHECK-LABEL: func.func @disabled
// CHECK: scf.for
// CHECK: memref.store
// CHECK-NOT: memref.store
// CHECK: return
func.func @disabled(%a: memref<16xindex>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  scf.for %i = %c0 to %c16 step %c1 {
    memref.store %i, %a[%i] : memref<16xindex>
  } {nounroll, unroll_count = 4 : i64}
  return
}

// CHECK-LABEL: func.func @no_hint
// CHECK: scf.for
// CHECK: memref.store
// CHECK-NOT: memref.store
// CHECK: return
func.func @no_hint(%a: memref<16xindex>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  scf.for %i = %c0 to %c16 step %c1 {
    memref.store %i, %a[%i] : memref<16xindex>
  }
  return
}

// The single remaining iteration is emitted as a scalar tail.
// CHECK-LABEL: func.func @remainder
// CHECK: scf.for
// CHECK: memref.store
// CHECK: memref.store
// CHECK: } {unroll_count
// CHECK-NOT: scf.for
// CHECK: memref.store
// CHECK: return
func.func @remainder(%a: memref<7xindex>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c7 = arith.constant 7 : index
  scf.for %i = %c0 to %c7 step %c1 {
    memref.store %i, %a[%i] : memref<7xindex>
  } {unroll_count = 2 : i64}
  return
}
