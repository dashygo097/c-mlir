// RUN: clang-to-mlir --disable-opt %s -- -std=c++17 | FileCheck %s --enable-var-scope

int explicit_cast(double value) { return static_cast<int>(value); }

// CHECK-LABEL: func.func @explicit_cast(%arg0: f64) -> i32
// CHECK: arith.fptosi %arg0 : f64 to i32

float c_style(unsigned value) { return (float)value; }

// CHECK-LABEL: func.func @c_style(%arg0: i32) -> f32
// CHECK: arith.uitofp %arg0 : i32 to f32

float implicit(unsigned value) { return value; }

// CHECK-LABEL: func.func @implicit(%arg0: i32) -> f32
// CHECK: arith.uitofp %arg0 : i32 to f32

const char *message(void) { return "ok"; }

// CHECK-LABEL: func.func @message() -> !llvm.ptr
// CHECK: %[[ADDRESS:.*]] = llvm.mlir.addressof
// CHECK: return %[[ADDRESS]] : !llvm.ptr

int *identity(int *value) { return value; }

// CHECK-LABEL: func.func @identity(%arg0: memref<?xi32, strided<[1], offset: ?>>) -> !llvm.ptr
// CHECK: memref.extract_aligned_pointer_as_index %arg0
// CHECK: return {{.*}} : !llvm.ptr
