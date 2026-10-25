// RUN: cmlir-opt %s --default-pipeline --verify-each | FileCheck %s --check-prefixes=OPT,UPDATE --enable-var-scope
// RUN: cmlir-opt %s --pass-pipeline='builtin.module(default-pipeline{inline=true})' --verify-each | FileCheck %s --check-prefixes=INLINE,UPDATE --enable-var-scope

// OPT-LABEL: func.func @local(
// OPT-NOT: memref.alloca
// OPT: %[[VALUE:.*]] = memref.load %arg0[%arg1]
// OPT: return %[[VALUE]] : i32
func.func @local(%data: memref<?xi32>, %index: index) -> i32 {
  %slot = memref.alloca() : memref<memref<?xi32>>
  memref.store %data, %slot[] : memref<memref<?xi32>>
  %pointer = memref.load %slot[] : memref<memref<?xi32>>
  %value = memref.load %pointer[%index] : memref<?xi32>
  return %value : i32
}

// OPT-LABEL: func.func @conditional(
// OPT-NOT: memref.alloca
// OPT: %[[POINTER:.*]] = arith.select %arg0, %arg2, %arg1 : memref<?xi32>
// OPT: %[[VALUE:.*]] = memref.load %[[POINTER]][%arg3]
// OPT: return %[[VALUE]] : i32
func.func @conditional(%condition: i1, %first: memref<?xi32>,
                       %second: memref<?xi32>, %index: index) -> i32 {
  %slot = memref.alloca() : memref<memref<?xi32>>
  memref.store %first, %slot[] : memref<memref<?xi32>>
  scf.if %condition {
    memref.store %second, %slot[] : memref<memref<?xi32>>
  }
  %pointer = memref.load %slot[] : memref<memref<?xi32>>
  %value = memref.load %pointer[%index] : memref<?xi32>
  return %value : i32
}

// OPT-LABEL: func.func @multiple_slots(
// OPT-NOT: memref.alloca
// OPT: %[[VALUE:.*]] = memref.load %arg0[%arg1]
// OPT: return %[[VALUE]] : i32
func.func @multiple_slots(%data: memref<?xi32>, %index: index) -> i32 {
  %inner = memref.alloca() : memref<memref<?xi32>>
  %outer = memref.alloca() : memref<memref<memref<?xi32>>>
  memref.store %data, %inner[] : memref<memref<?xi32>>
  memref.store %inner, %outer[] : memref<memref<memref<?xi32>>>
  %slot = memref.load %outer[] : memref<memref<memref<?xi32>>>
  %pointer = memref.load %slot[] : memref<memref<?xi32>>
  %value = memref.load %pointer[%index] : memref<?xi32>
  return %value : i32
}

func.func private @opaque(memref<memref<?xi32>>)

// OPT-LABEL: func.func @escaping(
// OPT: %[[SLOT:.*]] = memref.alloca() : memref<memref<?xi32>>
// OPT: call @opaque(%[[SLOT]])
// OPT: %[[POINTER:.*]] = memref.load %[[SLOT]][]
// OPT: memref.load %[[POINTER]]
func.func @escaping(%data: memref<?xi32>, %index: index) -> i32 {
  %slot = memref.alloca() : memref<memref<?xi32>>
  memref.store %data, %slot[] : memref<memref<?xi32>>
  func.call @opaque(%slot) : (memref<memref<?xi32>>) -> ()
  %pointer = memref.load %slot[] : memref<memref<?xi32>>
  %value = memref.load %pointer[%index] : memref<?xi32>
  return %value : i32
}

func.func private @set(%slot: memref<memref<?xi32>>, %replacement: memref<?xi32>) {
  memref.store %replacement, %slot[] : memref<memref<?xi32>>
  return
}

// OPT-LABEL: func.func @through_call(
// OPT: memref.alloca
// OPT: call @set
// INLINE-LABEL: func.func @through_call(
// INLINE-NOT: memref.alloca
// INLINE-NOT: call
// INLINE: %[[VALUE:.*]] = memref.load %arg1[%arg2]
// INLINE: return %[[VALUE]] : i32
func.func @through_call(%first: memref<?xi32>, %second: memref<?xi32>,
                        %index: index) -> i32 {
  %slot = memref.alloca() : memref<memref<?xi32>>
  memref.store %first, %slot[] : memref<memref<?xi32>>
  func.call @set(%slot, %second) : (memref<memref<?xi32>>, memref<?xi32>) -> ()
  %pointer = memref.load %slot[] : memref<memref<?xi32>>
  %value = memref.load %pointer[%index] : memref<?xi32>
  return %value : i32
}

// A mutable pointer slot becomes a loop-carried descriptor, preserving the
// updated offset and remaining extent without carrying its storage object.
// UPDATE-LABEL: func.func @advancing(
// UPDATE-NOT: memref.alloca
// UPDATE-NOT: memref.store
// UPDATE: %[[LOOP:.*]]:2 = scf.for
// UPDATE-SAME: iter_args(%{{.*}} = %{{.*}}, %[[POINTER:[^ ]+]] = %arg0)
// UPDATE: memref.load %[[POINTER]][
// UPDATE-NOT: memref.load
// UPDATE: %[[NEXT:.*]] = memref.reinterpret_cast
// UPDATE-NOT: memref.store
// UPDATE: scf.yield %{{.*}}, %[[NEXT]] : i32, memref<?xi32, strided<[1], offset: ?>>
// UPDATE-NEXT: }
// UPDATE-NEXT: return %[[LOOP]]#0 : i32
func.func @advancing(%data: memref<?xi32, strided<[1], offset: ?>>,
                     %count: index) -> i32 {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %zero = arith.constant 0 : i32
  %slot = memref.alloca() : memref<memref<?xi32, strided<[1], offset: ?>>>
  memref.store %data, %slot[] : memref<memref<?xi32, strided<[1], offset: ?>>>
  %sum = scf.for %i = %c0 to %count step %c1 iter_args(%acc = %zero) -> (i32) {
    %pointer = memref.load %slot[] : memref<memref<?xi32, strided<[1], offset: ?>>>
    %value = memref.load %pointer[%c0] : memref<?xi32, strided<[1], offset: ?>>
    %base, %offset, %size, %stride = memref.extract_strided_metadata %pointer
      : memref<?xi32, strided<[1], offset: ?>> -> memref<i32>, index, index, index
    %next_offset = arith.addi %offset, %c1 : index
    %remaining = arith.subi %size, %c1 : index
    %next = memref.reinterpret_cast %base to
      offset: [%next_offset], sizes: [%remaining], strides: [1]
      : memref<i32> to memref<?xi32, strided<[1], offset: ?>>
    memref.store %next, %slot[] : memref<memref<?xi32, strided<[1], offset: ?>>>
    %updated = arith.addi %acc, %value : i32
    scf.yield %updated : i32
  }
  return %sum : i32
}
