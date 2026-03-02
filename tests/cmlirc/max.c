// RUN: split-file %s %t
// RUN: cmlirc %t/max2.c | FileCheck %s --check-prefix=CHECK2
// RUN: cmlirc %t/max3.c | FileCheck %s --check-prefix=CHECK3

//--- max2.c
int max2(int a, int b) {
  return a > b ? a : b;
}

// CHECK2: func @max2(%arg0: i32, %arg1: i32) -> i32
// CHECK2: %[[V0:.*]] = arith.cmpi sgt, %arg0, %arg1 : i32
// CHECK2: %[[V1:.*]] = arith.select %[[V0]], %arg0, %arg1 : i32
// CHECK2: return %[[V1]] : i32

//--- max3.c
int max3(int a, int b, int c) {
  if (a > b && a > c) {
    return a;
  } else if (b > c) {
    return b;
  } else {
    return c;
  }
}

// CHECK3: func @max3(%arg0: i32, %arg1: i32, %arg2: i32) -> i32
// CHECK3-DAG: %[[false:.*]] = arith.constant false
// CHECK3: %[[V0:.*]] = arith.cmpi sgt, %arg0, %arg1 : i32
// CHECK3: %[[V1:.*]] = arith.cmpi sgt, %arg0, %arg2 : i32
// CHECK3: %[[V2:.*]] = arith.select %[[V0]], %[[V1]], %[[false]] : i1
// CHECK3: cf.cond_br %[[V2]], ^bb1(%arg0 : i32), ^bb2
// CHECK3: ^bb1(%[[V3:.*]]: i32):
// CHECK3: return %[[V3]] : i32
// CHECK3: ^bb2:
// CHECK3: %[[V4:.*]] = arith.cmpi sgt, %arg1, %arg2 : i32
// CHECK3: cf.cond_br %[[V4]], ^bb1(%arg1 : i32), ^bb1(%arg2 : i32)
