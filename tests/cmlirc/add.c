// RUN: cmlirc %s -function=add | FileCheck %s

int add(int a, int b) { return a + b; }

// CHECK: func @add(%arg0: i32, %arg1: i32) -> i32
// CHECK: %[[V0:.*]] = arith.addi %arg0, %arg1 : i32
// CHECK: return %[[V0]] : i32
