// RUN: clang-to-mlir %s -function=read -disable-opt -- -xc | FileCheck %s --implicit-check-not=llvm.

int read(int *pointer, int offset) { return *((pointer + offset + 1) - 1); }

// CHECK-LABEL: func.func @read
// CHECK: memref.extract_strided_metadata
// CHECK: arith.muli
// CHECK: memref.reinterpret_cast
// CHECK: memref.extract_strided_metadata
// CHECK: memref.reinterpret_cast
// CHECK: memref.extract_strided_metadata
// CHECK: memref.reinterpret_cast
// CHECK: memref.load
