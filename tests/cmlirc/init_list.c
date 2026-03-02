// RUN: split-file %s %t

// RUN: cmlirc %t/init_list0.c | FileCheck %s --check-prefix=CHECK0
// RUN: cmlirc %t/init_list1.c | FileCheck %s --check-prefix=CHECK1
// RUN: cmlirc %t/init_list2.c | FileCheck %s --check-prefix=CHECK2
// RUN: cmlirc %t/init_list3.c | FileCheck %s --check-prefix=CHECK3

//--- init_list0.c
int init_list() {
  int list[10];
  return list[0];
}

// CHECK0: func.func @init_list() -> i32
// CHECK0-DAG: %[[c0:.*]] = arith.constant 0 : index
// CHECK0: %[[V0:.*]] = memref.alloca() : memref<10xi32>
// CHECK0: %[[V1:.*]] = memref.load %[[V0]][%[[c0]]] : memref<10xi32>
// CHECK0: return %[[V1]] : i32

//--- init_list1.c
int init_list() {
  int list[10] = {0};
  return list[0];
}

// CHECK1: func.func @init_list() -> i32
// CHECK1-DAG: %[[c0:.*]] = arith.constant 0 : index
// CHECK1-DAG: %[[c0_i32:.*]] = arith.constant 0 : i32
// CHECK1: %[[V0:.*]] = memref.alloca() : memref<10xi32>
// CHECK1: affine.store %[[c0_i32]], %[[V0]][0] : memref<10xi32>
// CHECK1: %[[V1:.*]] = memref.load %[[V0]][%[[c0]]] : memref<10xi32>
// CHECK1: return %[[V1]] : i32

//--- init_list2.c
int init_list() {
  int list[10] = {0, 1, 2, 3, 4};
  return list[0];
}

// CHECK2: func.func @init_list() -> i32
// CHECK2-DAG: %[[c0:.*]] = arith.constant 0 : index
// CHECK2-DAG: %[[c0_i32:.*]] = arith.constant 0 : i32
// CHECK2-DAG: %[[c1_i32:.*]] = arith.constant 1 : i32
// CHECK2-DAG: %[[c2_i32:.*]] = arith.constant 2 : i32
// CHECK2-DAG: %[[c3_i32:.*]] = arith.constant 3 : i32
// CHECK2-DAG: %[[c4_i32:.*]] = arith.constant 4 : i32
// CHECK2: %[[V0:.*]] = memref.alloca() : memref<10xi32>
// CHECK2: affine.store %[[c0_i32]], %[[V0]][0] : memref<10xi32>
// CHECK2: affine.store %[[c1_i32]], %[[V0]][1] : memref<10xi32>
// CHECK2: affine.store %[[c2_i32]], %[[V0]][2] : memref<10xi32>
// CHECK2: affine.store %[[c3_i32]], %[[V0]][3] : memref<10xi32>
// CHECK2: affine.store %[[c4_i32]], %[[V0]][4] : memref<10xi32>
// CHECK2: %[[V1:.*]] = memref.load %[[V0]][%[[c0]]] : memref<10xi32>
// CHECK2: return %[[V1]] : i32

//--- init_list3.c
int init_list() {
  int list[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  return list[9];
}

// CHECK3: func.func @init_list() -> i32
// CHECK3-DAG: %[[c9:.*]] = arith.constant 9 : index
// CHECK3-DAG: %[[c0_i32:.*]] = arith.constant 0 : i32
// CHECK3-DAG: %[[c1_i32:.*]] = arith.constant 1 : i32
// CHECK3-DAG: %[[c2_i32:.*]] = arith.constant 2 : i32
// CHECK3-DAG: %[[c3_i32:.*]] = arith.constant 3 : i32
// CHECK3-DAG: %[[c4_i32:.*]] = arith.constant 4 : i32
// CHECK3-DAG: %[[c5_i32:.*]] = arith.constant 5 : i32
// CHECK3-DAG: %[[c6_i32:.*]] = arith.constant 6 : i32
// CHECK3-DAG: %[[c7_i32:.*]] = arith.constant 7 : i32
// CHECK3-DAG: %[[c8_i32:.*]] = arith.constant 8 : i32
// CHECK3-DAG: %[[c9_i32:.*]] = arith.constant 9 : i32
// CHECK3: %[[V0:.*]] = memref.alloca() : memref<10xi32>
// CHECK3: affine.store %[[c0_i32]], %[[V0]][0] : memref<10xi32>
// CHECK3: affine.store %[[c1_i32]], %[[V0]][1] : memref<10xi32>
// CHECK3: affine.store %[[c2_i32]], %[[V0]][2] : memref<10xi32>
// CHECK3: affine.store %[[c3_i32]], %[[V0]][3] : memref<10xi32>
// CHECK3: affine.store %[[c4_i32]], %[[V0]][4] : memref<10xi32>
// CHECK3: affine.store %[[c5_i32]], %[[V0]][5] : memref<10xi32>
// CHECK3: affine.store %[[c6_i32]], %[[V0]][6] : memref<10xi32>
// CHECK3: affine.store %[[c7_i32]], %[[V0]][7] : memref<10xi32>
// CHECK3: affine.store %[[c8_i32]], %[[V0]][8] : memref<10xi32>
// CHECK3: affine.store %[[c9_i32]], %[[V0]][9] : memref<10xi32>
// CHECK3: %[[V1:.*]] = memref.load %[[V0]][%[[c9]]] : memref<10xi32>
// CHECK3: return %[[V1]] : i32
