// RUN: cmlir-opt %s --loop-vectorize --verify-each | FileCheck %s

// Only the hinted loop is selected; restrict permits independent streams.
// CHECK-LABEL: func.func @selected
// CHECK: vector.transfer_read {{.*}} : memref<16xf32>, vector<4xf32>
// CHECK: arith.addf {{.*}} : vector<4xf32>
// CHECK: vector.transfer_write
// CHECK: scf.for
// CHECK: memref.load
// CHECK: memref.store
// CHECK-NOT: vector.
// CHECK: return
func.func @selected(%out: memref<16xf32> {cmlir.restrict}, %in: memref<16xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  scf.for %i = %c0 to %c16 step %c1 {
    %x = memref.load %in[%i] : memref<16xf32>
    %sum = arith.addf %x, %x : f32
    memref.store %sum, %out[%i] : memref<16xf32>
  } {vectorize, vectorize_width = 4 : i64}
  scf.for %i = %c0 to %c16 step %c1 {
    %x = memref.load %in[%i] : memref<16xf32>
    memref.store %x, %out[%i] : memref<16xf32>
  }
  return
}

// Unknown bounds need a full-vector path and a scalar tail.
// CHECK-LABEL: func.func @dynamic_tail
// CHECK: scf.for
// CHECK: scf.if
// CHECK: vector.transfer_read {{.*}} : memref<?xf32>, vector<8xf32>
// CHECK: vector.transfer_write
// CHECK: } else {
// CHECK: scf.for
// CHECK: memref.load
// CHECK: memref.store
// CHECK: return
func.func @dynamic_tail(%out: memref<?xf32> {cmlir.restrict}, %in: memref<?xf32>, %n: index) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  scf.for %i = %c0 to %n step %c1 {
    %x = memref.load %in[%i] : memref<?xf32>
    memref.store %x, %out[%i] : memref<?xf32>
  } {vectorize, vectorize_width = 8 : i64}
  return
}

// Potential overlap must not introduce cross-iteration memory changes.
// CHECK-LABEL: func.func @may_alias
// CHECK-NOT: vector.
// CHECK: memref.load
// CHECK: memref.store
// CHECK-NOT: vector.
// CHECK: return
func.func @may_alias(%out: memref<16xf32>, %in: memref<16xf32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  scf.for %i = %c0 to %c16 step %c1 {
    %x = memref.load %in[%i] : memref<16xf32>
    memref.store %x, %out[%i] : memref<16xf32>
  } {vectorize, vectorize_width = 4 : i64}
  return
}

// A loop-carried memory dependence remains scalar even with restrict.
// CHECK-LABEL: func.func @dependent
// CHECK-NOT: vector.
// CHECK: memref.load
// CHECK: memref.store
// CHECK-NOT: vector.
// CHECK: return
func.func @dependent(%a: memref<16xf32> {cmlir.restrict}) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  scf.for %i = %c1 to %c16 step %c1 {
    %prev = arith.subi %i, %c1 : index
    %x = memref.load %a[%prev] : memref<16xf32>
    memref.store %x, %a[%i] : memref<16xf32>
  } {vectorize, vectorize_width = 4 : i64}
  return
}

// Strict FP accumulation must keep the original evaluation order.
// CHECK-LABEL: func.func @strict_reduction
// CHECK-NOT: vector.
// CHECK: scf.for
// CHECK: arith.addf {{.*}} : f32
// CHECK-NOT: vector.
// CHECK: return
func.func @strict_reduction(%a: memref<16xf32>) -> f32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  %zero = arith.constant 0.0 : f32
  %sum = scf.for %i = %c0 to %c16 step %c1 iter_args(%acc = %zero) -> f32 {
    %x = memref.load %a[%i] : memref<16xf32>
    %next = arith.addf %acc, %x : f32
    scf.yield %next : f32
  } {vectorize, vectorize_width = 4 : i64}
  return %sum : f32
}

// Explicit reassociation permission enables the reduction transformation.
// CHECK-LABEL: func.func @reassociated_reduction
// CHECK: vector.transfer_read
// CHECK: vector.reduction
// CHECK: return
func.func @reassociated_reduction(%a: memref<16xf32>) -> f32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  %zero = arith.constant 0.0 : f32
  %sum = scf.for %i = %c0 to %c16 step %c1 iter_args(%acc = %zero) -> f32 {
    %x = memref.load %a[%i] : memref<16xf32>
    %next = arith.addf %acc, %x fastmath<reassoc> : f32
    scf.yield %next : f32
  } {vectorize, vectorize_width = 4 : i64}
  return %sum : f32
}

// A partial final vector must not contribute out-of-range elements.
// CHECK-LABEL: func.func @reduction_tail(
// CHECK: scf.for
// CHECK: vector.create_mask {{.*}} : vector<4xi1>
// CHECK: vector.transfer_read {{.*}} : memref<7xi32>, vector<4xi32>
// CHECK: arith.select {{.*}} : vector<4xi1>, vector<4xi32>
// CHECK: arith.addi {{.*}} : vector<4xi32>
// CHECK: %[[SUM:.*]] = vector.reduction <add>, {{.*}} : vector<4xi32> into i32
// CHECK: %[[RESULT:.*]] = arith.addi %[[SUM]], %arg1 : i32
// CHECK: return %[[RESULT]] : i32
func.func @reduction_tail(%a: memref<7xi32>, %initial: i32) -> i32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c7 = arith.constant 7 : index
  %sum = scf.for %i = %c0 to %c7 step %c1 iter_args(%acc = %initial) -> i32 {
    %x = memref.load %a[%i] : memref<7xi32>
    %next = arith.addi %acc, %x : i32
    scf.yield %next : i32
  } {vectorize, vectorize_width = 4 : i64}
  return %sum : i32
}

// Dynamic reductions share the tail mask and retain their own initial values.
// CHECK-LABEL: func.func @dynamic_reductions(
// CHECK: %[[LOOP:.*]]:2 = scf.for
// CHECK: %[[MASK:.*]] = vector.create_mask {{.*}} : vector<4xi1>
// CHECK: vector.transfer_read {{.*}} : memref<?xi32>, vector<4xi32>
// CHECK: arith.select %[[MASK]]
// CHECK: arith.select %[[MASK]]
// CHECK: %[[SUM:.*]] = vector.reduction <add>, %[[LOOP]]#0 : vector<4xi32> into i32
// CHECK: %[[SUM_RESULT:.*]] = arith.addi %[[SUM]], %arg2 : i32
// CHECK: %[[XOR:.*]] = vector.reduction <xor>, %[[LOOP]]#1 : vector<4xi32> into i32
// CHECK: %[[XOR_RESULT:.*]] = arith.xori %[[XOR]], %arg3 : i32
// CHECK: return %[[SUM_RESULT]], %[[XOR_RESULT]] : i32, i32
func.func @dynamic_reductions(%a: memref<?xi32>, %n: index, %sum_initial: i32, %xor_initial: i32) -> (i32, i32) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %result:2 = scf.for %i = %c0 to %n step %c1
      iter_args(%sum = %sum_initial, %xor = %xor_initial) -> (i32, i32) {
    %x = memref.load %a[%i] : memref<?xi32>
    %next_sum = arith.addi %sum, %x : i32
    %next_xor = arith.xori %xor, %x : i32
    scf.yield %next_sum, %next_xor : i32, i32
  } {vectorize, vectorize_width = 4 : i64}
  return %result#0, %result#1 : i32, i32
}

// Width one is consumed without requesting vectorization.
// CHECK-LABEL: func.func @width_one
// CHECK-NOT: vector.
// CHECK: scf.for
// CHECK: memref.store
// CHECK-NOT: vectorize
// CHECK: return
func.func @width_one(%a: memref<16xindex>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  scf.for %i = %c0 to %c16 step %c1 {
    memref.store %i, %a[%i] : memref<16xindex>
  } {vectorize, vectorize_width = 1 : i64}
  return
}
