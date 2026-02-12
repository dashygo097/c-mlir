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
// CHECK: [[V0:.*]] = arith.cmpi sgt, %arg0, %arg1 : i32
// CHECK: [[V1:.*]] = scf.if [[V0:.*]] -> (i32) {
// CHECK:   [[V2:.*]] = arith.cmpi sgt, %arg0, %arg2 : i32
// CHECK:   [[V3:.*]] = arith.select [[V2:.*]], %arg0, %arg2 : i32
// CHECK:   scf.yield [[V3:.*]] : i32
// CHECK: } else {
// CHECK:   [[V2:.*]] = arith.cmpi sgt, %arg1, %arg2 : i32
// CHECK:   [[V3:.*]] = arith.select [[V2:.*]], %arg1, %arg2 : i32
// CHECK:   scf.yield [[V3:.*]] : i32
// CHECK: }
// CHECK: return [[V1:.*]] : i32
