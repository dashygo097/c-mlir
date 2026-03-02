// RUN: cmlirc %s -function=max3 | FileCheck %s

int max3(int a, int b, int c) {
  if (a > b)
    if (a > c)
      return a;
    else
      return c;
  else if (b > c)
    return b;
  else
    return c;
}

// CHECK: func.func @max3(%arg0: i32, %arg1: i32, %arg2: i32) -> i32
// CHECK: %[[V0:.*]] = arith.cmpi sgt, %arg0, %arg1 : i32
// CHECK: %[[V1:.*]] = arith.select %[[V0]], %arg0, %arg1 : i32
// CHECK: %[[V2:.*]] = arith.cmpi sgt, %[[V1]], %arg2 : i32
// CHECK: %[[V3:.*]] = arith.select %[[V2]], %[[V1]], %arg2 : i32
// CHECK: return [[V3:.*]] : i32
