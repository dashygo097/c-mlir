// RUN: clang-to-mlir %s -function=read -disable-opt -- -xc | FileCheck %s --implicit-check-not=llvm.
// RUN: clang-to-mlir %s -function=read -- -xc | FileCheck %s --check-prefix=CLEAN --implicit-check-not=llvm.

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

// CLEAN-LABEL: func.func @read
// CLEAN: memref.extract_strided_metadata
// CLEAN-NOT: memref.extract_strided_metadata
// CLEAN: memref.reinterpret_cast
// CLEAN-NOT: memref.extract_strided_metadata
// CLEAN-NOT: memref.reinterpret_cast
// CLEAN: memref.load
// CLEAN-NOT: memref.reinterpret_cast
