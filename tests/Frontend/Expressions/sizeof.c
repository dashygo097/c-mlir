// RUN: clang-to-mlir --disable-opt %s -- | FileCheck %s

int _sizeof(int a) {
  return sizeof(a);
}

// CHECK-LABEL: func.func @_sizeof(%arg0: i32) -> i32
// CHECK: %[[SIZE:.*]] = arith.constant 4 : i64
// CHECK: %[[RESULT:.*]] = arith.trunci %[[SIZE]] : i64 to i32
// CHECK: return %[[RESULT]] : i32
