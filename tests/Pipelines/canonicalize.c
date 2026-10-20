// RUN: clang-to-mlir %s --function=max2 -- | FileCheck %s --check-prefix=CHECK2
// RUN: clang-to-mlir --disable-opt %s --function=max2 -- | cmlir-opt --pass-pipeline='builtin.module(default-pipeline)' | FileCheck %s --check-prefix=CHECK2
// RUN: clang-to-mlir %s --function=max4 -- | FileCheck %s --check-prefix=CHECK4
// RUN: clang-to-mlir --disable-opt %s --function=max4 -- | cmlir-opt --pass-pipeline='builtin.module(default-pipeline)' | FileCheck %s --check-prefix=CHECK4

int max2(int a, int b) {
  return a > b ? a : b;
}

// CHECK2: func.func @max2(%arg0: i32, %arg1: i32) -> i32
// CHECK2: %[[V0:.*]] = arith.maxsi %arg0, %arg1 : i32
// CHECK2: return %[[V0]] : i32

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
