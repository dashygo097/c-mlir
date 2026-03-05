// RUN: cmlirc %s | FileCheck %s

int _sizeof(int a) {
  return sizeof(a);
}

// CHECK: func.func @_sizeof(%arg0: i32) -> i32
// CHECK: %[[c4_i32:.*]] = arith.constant 4 : i32
// CHECK: return %[[c4_i32]] : i32
