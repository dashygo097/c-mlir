// RUN: cmlirc %s | FileCheck %s

int pre_decrement(int a) { return --a; }
int post_decrement(int a) { return a--; }

// CHECK: func.func @pre_decrement(%arg0: i32) -> i32
// CHECK: %c1_i32 = arith.constant 1 : i32
// CHECK: %[[V0:.*]] = arith.subi %arg0, %c1_i32 : i32
// CHECK: return %[[V0]] : i32
// CHECK: func.func @post_decrement(%arg0: i32) -> i32
// CHECK: %c1_i32 = arith.constant 1 : i32
// CHECK: %[[V0:.*]] = arith.subi %arg0, %c1_i32 : i32
// CHECK: return %arg0 : i32
