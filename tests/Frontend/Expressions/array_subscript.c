// RUN: clang-to-mlir --disable-opt %s -- | FileCheck %s --enable-var-scope

int read(int *array, int index) { return array[index]; }

// CHECK-LABEL: func.func @read(%arg0: memref<?xi32>, %arg1: i32) -> i32
// CHECK: %[[WIDE:.*]] = arith.extsi %arg1 : i32 to i64
// CHECK: %[[I:.*]] = arith.index_cast %[[WIDE]] : i64 to index
// CHECK: %[[R:.*]] = memref.load %arg0[%[[I]]]
// CHECK: return %[[R]] : i32

void write(int *array, int index, int value) { array[index] = value; }

// CHECK-LABEL: func.func @write
// CHECK: %[[SLOT:.*]] = memref.alloca() : memref<i32>
// CHECK: memref.store %arg1, %[[SLOT]][]
// CHECK: %[[INDEX:.*]] = memref.load %[[SLOT]][]
// CHECK: %[[WIDE:.*]] = arith.extsi %[[INDEX]] : i32 to i64
// CHECK: %[[I:.*]] = arith.index_cast %[[WIDE]] : i64 to index
// CHECK: memref.store %arg2, %arg0[%[[I]]]
// CHECK: return
