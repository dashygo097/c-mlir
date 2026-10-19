// RUN: clang-to-mlir --disable-opt %s -- | FileCheck %s --enable-var-scope

int pre_increment(int value) { return ++value; }

// CHECK-LABEL: func.func @pre_increment
// CHECK: %[[OLD:.*]] = memref.load
// CHECK: %[[NEW:.*]] = arith.addi %[[OLD]], {{.*}}
// CHECK: memref.store %[[NEW]]
// CHECK: return %[[NEW]]

int post_increment(int value) { return value++; }

// CHECK-LABEL: func.func @post_increment
// CHECK: %[[OLD:.*]] = memref.load
// CHECK: %[[NEW:.*]] = arith.addi %[[OLD]], {{.*}}
// CHECK: memref.store %[[NEW]]
// CHECK: return %[[OLD]]

int pre_decrement(int value) { return --value; }

// CHECK-LABEL: func.func @pre_decrement
// CHECK: %[[OLD:.*]] = memref.load
// CHECK: %[[NEW:.*]] = arith.subi %[[OLD]], {{.*}}
// CHECK: memref.store %[[NEW]]
// CHECK: return %[[NEW]]

int post_decrement(int value) { return value--; }

// CHECK-LABEL: func.func @post_decrement
// CHECK: %[[OLD:.*]] = memref.load
// CHECK: %[[NEW:.*]] = arith.subi %[[OLD]], {{.*}}
// CHECK: memref.store %[[NEW]]
// CHECK: return %[[OLD]]
