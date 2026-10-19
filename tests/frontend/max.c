// RUN: split-file %s %t
// RUN: clang-to-mlir %t/max2.c | FileCheck %s --check-prefix=CHECK2
// RUN: clang-to-mlir %t/max3.c | FileCheck %s --check-prefix=CHECK3 --implicit-check-not=cf.cond_br --implicit-check-not=cf.br
// RUN: clang-to-mlir %t/max3.c --lift-cf-to-scf=false | FileCheck %s --check-prefix=CFG
// RUN: clang-to-mlir %t/max3.c --disable-opt | FileCheck %s --check-prefix=CFG
// RUN: clang-to-mlir %t/max4.c | FileCheck %s --check-prefix=CHECK4

//--- max2.c
int max2(int a, int b) {
  return a > b ? a : b;
}

// CHECK2: func.func @max2(%arg0: i32, %arg1: i32) -> i32
// CHECK2: %[[V0:.*]] = arith.maxsi %arg0, %arg1 : i32
// CHECK2: return %[[V0]] : i32

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

// CHECK3: func.func @max3(%arg0: i32, %arg1: i32, %arg2: i32) -> i32
// CHECK3-DAG: %[[false:.*]] = arith.constant false
// CHECK3: %[[V0:.*]] = arith.cmpi sgt, %arg0, %arg1 : i32
// CHECK3: %[[V2:.*]] = scf.if %[[V0]] -> (i1) {
// CHECK3: %[[V1:.*]] = arith.cmpi sgt, %arg0, %arg2 : i32
// CHECK3: scf.yield %[[V1]] : i1
// CHECK3: } else {
// CHECK3: scf.yield %[[false]] : i1
// CHECK3: }
// CHECK3: %[[RESULT:.*]] = scf.if %[[V2]] -> (i32) {
// CHECK3: scf.yield %arg0 : i32
// CHECK3: } else {
// CHECK3: %[[MAX_BC:.*]] = arith.maxsi %arg1, %arg2 : i32
// CHECK3: scf.yield %[[MAX_BC]] : i32
// CHECK3: }
// CHECK3: return %[[RESULT]] : i32

// CFG-LABEL: func.func @max3
// CFG: cf.cond_br

//--- max4.c
int max4(int a, int b, int c, int d) {
  int max = a;
  if (b > max) {
    max = b;
  }
  if (c > max) {
    max = c;
  }
  if (d > max) {
    return d;
  }
  return max;
}

// CHECK4: func.func @max4(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: i32) -> i32
// CHECK4: %[[V0:.*]] = arith.maxsi %arg1, %arg0 : i32
// CHECK4: %[[V1:.*]] = arith.maxsi %arg2, %[[V0]] : i32
// CHECK4: %[[V2:.*]] = arith.maxsi %arg3, %[[V1]] : i32
// CHECK4: return %[[V2]] : i32
