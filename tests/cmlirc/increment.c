// RUN: cmlirc %s | FileCheck %s

int pre_increment(int a) { return ++a; }
int post_increment(int a) { return a++; }

// CHECK: func.func @pre_increment(%arg0: i32) -> i32
// CHECK: %c1_i32 = arith.constant 1 : i32
// CHECK: %[[V0:.*]] = arith.addi %arg0, %c1_i32 : i32
// CHECK: return %[[V0]] : i32
// CHECK: func.func @post_increment(%arg0: i32) -> i32
// CHECK: %c1_i32 = arith.constant 1 : i32
// CHECK: %[[V0:.*]] = arith.addi %arg0, %c1_i32 : i32
// CHECK: return %arg0 : i32
