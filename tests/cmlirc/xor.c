// RUN: cmlirc %s -function=_xor | FileCheck %s

int _xor(int a, int b) {
  return a ^ b;
}

// CHECK: func.func @_xor(%arg0: i32, %arg1: i32) -> i32
// CHECK: %[[V0:.*]] = arith.xori %arg0, %arg1 : i32
// CHECK: return %[[V0]] : i32
