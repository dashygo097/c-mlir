// RUN: clang-to-mlir --disable-opt %s -- -xc | FileCheck %s --enable-var-scope --implicit-check-not=llvm.

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

// CHECK-LABEL: func.func @compound
// CHECK: %[[SLOT:.*]] = memref.alloca() : memref<memref<?xi32
// CHECK: %[[OLD:.*]] = memref.load %[[SLOT]][]
// CHECK: memref.extract_strided_metadata %[[OLD]]
// CHECK: %[[ADD:.*]] = memref.reinterpret_cast
// CHECK: memref.store %[[ADD]], %[[SLOT]][]
// CHECK: memref.extract_strided_metadata
// CHECK: arith.subi
// CHECK: %[[SUB:.*]] = memref.reinterpret_cast
// CHECK: memref.store %[[SUB]], %[[SLOT]][]
int compound(int *pointer, int offset) {
  pointer += offset;
  pointer -= 1;
  return *pointer;
}

// CHECK-LABEL: func.func @post_increment
// CHECK: %[[OLD:.*]] = memref.load %{{.*}}[] : memref<memref<?xi32
// CHECK: memref.extract_strided_metadata %[[OLD]]
// CHECK: %[[NEXT:.*]] = memref.reinterpret_cast
// CHECK: memref.store %[[NEXT]]
// CHECK: memref.load %[[OLD]][
int post_increment(int *pointer) { return *pointer++; }

// CHECK-LABEL: func.func @pre_decrement
// CHECK: arith.subi
// CHECK: %[[NEXT:.*]] = memref.reinterpret_cast
// CHECK: memref.store %[[NEXT]]
// CHECK: memref.load %[[NEXT]][
int pre_decrement(int *pointer) { return *--pointer; }
