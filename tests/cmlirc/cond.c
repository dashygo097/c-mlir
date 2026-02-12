// RUN: cmlirc %s -function=max2 | FileCheck %s

int max2(int a, int b) {
  if (a > b) {
    return a;
  } else {
    return b;
  }
}

// CHECK: func @max2(%arg0: i32, %arg1: i32) -> i32
// CHECK: %[[V0:.*]] = arith.cmpi sgt, %arg0, %arg1 : i32
// CHECK: %[[V1:.*]] = arith.select %[[V0]], %arg0, %arg1 : i32
// CHECK: return %[[V1]] : i32
