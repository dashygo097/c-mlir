// RUN: clang-to-mlir %s -- | FileCheck %s --implicit-check-not=llvm.

int read(int *pointer, int offset) { return *((pointer + offset + 1) - 1); }

// CHECK-LABEL: func.func @read
// CHECK: memref.extract_strided_metadata
// CHECK-NOT: memref.extract_strided_metadata
// CHECK: memref.reinterpret_cast
// CHECK-NOT: memref.extract_strided_metadata
// CHECK-NOT: memref.reinterpret_cast
// CHECK: memref.load
// CHECK-NOT: memref.reinterpret_cast
