// RUN: cmlir-opt %s --struct2memref | FileCheck %s

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
