// RUN: clang-to-mlir --disable-opt %s -- | FileCheck %s --enable-var-scope

int uninitialized(void) {
  int a[2];
  return a[0];
}

// CHECK-LABEL: func.func @uninitialized
// CHECK: %[[A:.*]] = memref.alloca() : memref<2xi32>
// CHECK-NOT: memref.store
// CHECK: memref.load %[[A]]

int zero(void) {
  int a[2] = {0};
  return a[0];
}

// CHECK-LABEL: func.func @zero
// CHECK: arith.constant 0 : i32
// CHECK-COUNT-2: memref.store
// CHECK: memref.load

int partial(void) {
  int a[3] = {1, 2};
  return a[2];
}

// CHECK-LABEL: func.func @partial
// CHECK-COUNT-3: memref.store
// CHECK: memref.load

int full(void) {
  int a[3] = {1, 2, 3};
  return a[2];
}

// CHECK-LABEL: func.func @full
// CHECK-COUNT-3: memref.store
// CHECK: memref.load

int multidimensional(void) {
  int a[2][2] = {{1, 2}, {3, 4}};
  return a[1][1];
}

// CHECK-LABEL: func.func @multidimensional
// CHECK: memref.alloca() : memref<2x2xi32>
// CHECK-COUNT-4: memref.store
// CHECK: memref.load

float floating(void) {
  float a[2][2] = {{0.1f, 0.2f}, {0.3f, 0.4f}};
  return a[1][1];
}

// CHECK-LABEL: func.func @floating
// CHECK: memref.alloca() : memref<2x2xf32>
// CHECK-COUNT-4: memref.store
// CHECK: memref.load
