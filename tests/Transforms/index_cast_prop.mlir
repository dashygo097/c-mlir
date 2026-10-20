// RUN: cmlir-opt %s --split-input-file --arith-cast-prop | FileCheck %s

// An i32 -> index -> i32 round trip is exact with a 64-bit index.
module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 64 : i64>>} {
  // CHECK-LABEL: func.func @signed_widen
  // CHECK-NOT: arith.index_cast
  // CHECK: return %arg0 : i32
  func.func @signed_widen(%x: i32) -> i32 {
    %i = arith.index_cast %x : i32 to index
    %r = arith.index_cast %i : index to i32
    return %r : i32
  }

  // CHECK-LABEL: func.func @unsigned_widen
  // CHECK-NOT: arith.index_castui
  // CHECK: return %arg0 : i32
  func.func @unsigned_widen(%x: i32) -> i32 {
    %i = arith.index_castui %x : i32 to index
    %r = arith.index_castui %i : index to i32
    return %r : i32
  }

  // Unknown index values may lose bits when narrowed to i32.
  // CHECK-LABEL: func.func @unknown_narrow
  // CHECK: arith.index_cast
  // CHECK: arith.index_cast
  // CHECK: return
  func.func @unknown_narrow(%x: index) -> index {
    %i = arith.index_cast %x : index to i32
    %r = arith.index_cast %i : i32 to index
    return %r : index
  }

  // CHECK-LABEL: func.func @unsigned_unknown_narrow
  // CHECK: arith.index_castui
  // CHECK: arith.index_castui
  // CHECK: return
  func.func @unsigned_unknown_narrow(%x: index) -> index {
    %i = arith.index_castui %x : index to i32
    %r = arith.index_castui %i : i32 to index
    return %r : index
  }

  // Range analysis proves loop indices fit in i32 even on a 64-bit target.
  // CHECK-LABEL: func.func @bounded_index
  // CHECK: scf.for %[[I:.*]] =
  // CHECK-NOT: arith.index_cast
  // CHECK: memref.store %[[I]]
  func.func @bounded_index(%out: memref<16xindex>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    scf.for %i = %c0 to %c16 step %c1 {
      %n = arith.index_cast %i : index to i32
      %r = arith.index_cast %n : i32 to index
      memref.store %r, %out[%i] : memref<16xindex>
    }
    return
  }
}

// -----

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<index, 32 : i64>>} {
  // Equal widths make this exact without a range proof.
  // CHECK-LABEL: func.func @index32
  // CHECK-NOT: arith.index_cast
  // CHECK: return %arg0 : index
  func.func @index32(%x: index) -> index {
    %i = arith.index_cast %x : index to i32
    %r = arith.index_cast %i : i32 to index
    return %r : index
  }

  // A 32-bit index cannot preserve an arbitrary i64.
  // CHECK-LABEL: func.func @integer64
  // CHECK: arith.index_cast
  // CHECK: arith.index_cast
  // CHECK: return
  func.func @integer64(%x: i64) -> i64 {
    %i = arith.index_cast %x : i64 to index
    %r = arith.index_cast %i : index to i64
    return %r : i64
  }
}
