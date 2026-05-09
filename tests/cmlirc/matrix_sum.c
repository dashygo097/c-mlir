// RUN: cmlirc %s -function=matrix_sum --raise-memref-to-affine | FileCheck %s

int matrix_sum() {
  int mat[2][3] = {
      {1, 2, 3},
      {4, 5, 6}};

  return mat[0][0] + mat[0][1] + mat[0][2] +
         mat[1][0] + mat[1][1] + mat[1][2];
}

// CHECK: func.func @matrix_sum() -> i32
// CHECK-DAG: %[[c21_i32:.*]] = arith.constant 21 : i32
// CHECK: return %[[c21_i32]] : i32
