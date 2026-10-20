// RUN: split-file %s %t
// RUN: clang-to-mlir %t/matrix_sum.c --affine-raise-from-memref -- | FileCheck %s --check-prefix=MATRIX
// RUN: clang-to-mlir --disable-opt %t/matrix_sum.c -- | cmlir-opt --pass-pipeline='builtin.module(default-pipeline{affine-raise-from-memref=true})' | FileCheck %s --check-prefix=MATRIX
// RUN: clang-to-mlir %t/init_list0.c --affine-raise-from-memref -- | FileCheck %s --check-prefix=CHECK0
// RUN: clang-to-mlir --disable-opt %t/init_list0.c -- | cmlir-opt --pass-pipeline='builtin.module(default-pipeline{affine-raise-from-memref=true})' | FileCheck %s --check-prefix=CHECK0
// RUN: clang-to-mlir %t/init_list1.c --affine-raise-from-memref -- | FileCheck %s --check-prefix=CHECK1
// RUN: clang-to-mlir --disable-opt %t/init_list1.c -- | cmlir-opt --pass-pipeline='builtin.module(default-pipeline{affine-raise-from-memref=true})' | FileCheck %s --check-prefix=CHECK1
// RUN: clang-to-mlir %t/init_list2.c --affine-raise-from-memref -- | FileCheck %s --check-prefix=CHECK2
// RUN: clang-to-mlir --disable-opt %t/init_list2.c -- | cmlir-opt --pass-pipeline='builtin.module(default-pipeline{affine-raise-from-memref=true})' | FileCheck %s --check-prefix=CHECK2
// RUN: clang-to-mlir %t/init_list3.c --affine-raise-from-memref -- | FileCheck %s --check-prefix=CHECK3
// RUN: clang-to-mlir --disable-opt %t/init_list3.c -- | cmlir-opt --pass-pipeline='builtin.module(default-pipeline{affine-raise-from-memref=true})' | FileCheck %s --check-prefix=CHECK3

//--- matrix_sum.c
int matrix_sum() {
  int mat[2][3] = {
      {1, 2, 3},
      {4, 5, 6}};

  return mat[0][0] + mat[0][1] + mat[0][2] +
         mat[1][0] + mat[1][1] + mat[1][2];
}

// MATRIX: func.func @matrix_sum() -> i32
// MATRIX-DAG: %[[c21_i32:.*]] = arith.constant 21 : i32
// MATRIX: return %[[c21_i32]] : i32

//--- init_list0.c
int init_list() {
  int list[10];
  return list[0];
}

// CHECK0: func.func @init_list() -> i32
// CHECK0: %[[UNINITIALIZED:.*]] = ub.poison : i32
// CHECK0: return %[[UNINITIALIZED]] : i32

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
