// RUN: split-file %s %t

// RUN: cmlirc %t/init_list0.c --raise-memref-to-affine | FileCheck %s --check-prefix=CHECK0
// RUN: cmlirc %t/init_list1.c --raise-memref-to-affine | FileCheck %s --check-prefix=CHECK1
// RUN: cmlirc %t/init_list2.c --raise-memref-to-affine | FileCheck %s --check-prefix=CHECK2
// RUN: cmlirc %t/init_list3.c --raise-memref-to-affine | FileCheck %s --check-prefix=CHECK3

//--- init_list0.c
int init_list() {
  int list[10];
  return list[0];
}

// CHECK0: func.func @init_list() -> i32
// CHECK0: %[[V0:.*]] = memref.alloca() : memref<10xi32>
// CHECK0: %[[V1:.*]] = affine.load %[[V0]][0] : memref<10xi32>
// CHECK0: return %[[V1]] : i32

//--- init_list1.c
int init_list() {
  int list[10] = {0};
  return list[0];
}

// CHECK1: func.func @init_list() -> i32
// CHECK1-DAG: %[[c0_i32:.*]] = arith.constant 0 : i32
// CHECK1: return %[[c0_i32]] : i32

//--- init_list2.c
int init_list() {
  int list[10] = {0, 1, 2, 3, 4};
  return list[4];
}

// CHECK2: func.func @init_list() -> i32
// CHECK2-DAG: %[[c4_i32:.*]] = arith.constant 4 : i32
// CHECK2: return %[[c4_i32]] : i32

//--- init_list3.c
int init_list() {
  int list[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  return list[9];
}

// CHECK3: func.func @init_list() -> i32
// CHECK3-DAG: %[[c9_i32:.*]] = arith.constant 9 : i32
// CHECK3: return %[[c9_i32]] : i32
