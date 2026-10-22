// RUN: cmlir-opt %s --struct-to-memref | FileCheck %s

// CHECK-LABEL: func.func @homogeneous
// CHECK-SAME: %[[ARG:.*]]: memref<?x2xf32>
// CHECK-NOT: llvm.extractvalue
// CHECK: %[[FIELD:.*]] = memref.load %[[ARG]]
// CHECK: return %[[FIELD]] : f32
func.func @homogeneous(%point: !llvm.struct<(f32, f32)>) -> f32 {
  %field = llvm.extractvalue %point[1] : !llvm.struct<(f32, f32)>
  return %field : f32
}

// Mixed layouts do not fit the homogeneous memref representation.
// CHECK-LABEL: func.func @mixed
// CHECK-SAME: !llvm.struct<(f32, i32)>
// CHECK: llvm.extractvalue
// CHECK: return
func.func @mixed(%point: !llvm.struct<(f32, i32)>) -> f32 {
  %field = llvm.extractvalue %point[0] : !llvm.struct<(f32, i32)>
  return %field : f32
}

// CHECK-LABEL: func.func @pointer_field
// CHECK-SAME: %[[PTR:.*]]: memref<?x2xf32>
// CHECK-NOT: llvm.getelementptr
// CHECK-NOT: llvm.load
// CHECK: memref.load %[[PTR]]
// CHECK: return
func.func @pointer_field(%point: !llvm.ptr) -> f32 {
  %field = llvm.getelementptr %point[0, 1]
      : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(f32, f32)>
  %value = llvm.load %field : !llvm.ptr -> f32
  return %value : f32
}

// An aggregate initializer must survive when whole-object conversion is not
// supported. Keeping the LLVM storage is preferable to dropping its value.
// CHECK-LABEL: func.func @initialized
// CHECK-SAME: %[[POINT:.*]]: !llvm.struct<(f32, f32)>
// CHECK: %[[SLOT:.*]] = llvm.alloca
// CHECK: llvm.store %[[POINT]], %[[SLOT]]
// CHECK: llvm.load
func.func @initialized(%point: !llvm.struct<(f32, f32)>) -> f32 {
  %one = arith.constant 1 : i64
  %slot = llvm.alloca %one x !llvm.struct<(f32, f32)> : (i64) -> !llvm.ptr
  llvm.store %point, %slot : !llvm.struct<(f32, f32)>, !llvm.ptr
  %field = llvm.getelementptr %slot[0, 1]
      : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(f32, f32)>
  %value = llvm.load %field : !llvm.ptr -> f32
  return %value : f32
}

// CHECK-LABEL: func.func @dynamic
// CHECK: %[[N:.*]] = arith.index_cast %arg0 : i64 to index
// CHECK: memref.alloca(%[[N]]) : memref<?x2xf32>
// CHECK-NOT: memref.alloc(
// CHECK: memref.store
func.func @dynamic(%count: i64) {
  %slot = llvm.alloca %count x !llvm.struct<(f32, f32)> : (i64) -> !llvm.ptr
  %field = llvm.getelementptr %slot[0, 1]
      : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(f32, f32)>
  %one = arith.constant 1.0 : f32
  llvm.store %one, %field : f32, !llvm.ptr
  return
}

// Function calls must keep matching signatures until call conversion exists.
// CHECK-LABEL: func.func @callee
// CHECK-SAME: !llvm.struct<(f32, f32)>
func.func @callee(%point: !llvm.struct<(f32, f32)>) -> f32 {
  %value = llvm.extractvalue %point[0] : !llvm.struct<(f32, f32)>
  return %value : f32
}

// CHECK-LABEL: func.func @caller
// CHECK-SAME: !llvm.struct<(f32, f32)>
// CHECK: call @callee({{.*}}) : (!llvm.struct<(f32, f32)>) -> f32
func.func @caller(%point: !llvm.struct<(f32, f32)>) -> f32 {
  %value = func.call @callee(%point) : (!llvm.struct<(f32, f32)>) -> f32
  return %value : f32
}
