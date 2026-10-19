// RUN: clang-to-mlir --disable-opt %s -- | FileCheck %s --enable-var-scope

int forward(int);

// CHECK-LABEL: func.func @forward(%arg0: i32) -> i32
// CHECK: return %arg0 : i32

int callee(int a, int b) { return a + b; }

// CHECK-LABEL: func.func @callee(%arg0: i32, %arg1: i32) -> i32
// CHECK: arith.addi %arg0, %arg1

int caller(void) { return callee(1, forward(2)); }

// CHECK-LABEL: func.func @caller() -> i32
// CHECK: %[[FORWARD:.*]] = call @forward
// CHECK: %[[RESULT:.*]] = call @callee({{.*}}, %[[FORWARD]])
// CHECK: return %[[RESULT]] : i32

int forward(int value) { return value; }

int recursive(int n) {
  if (n <= 0) {
    return 0;
  }
  return 1 + recursive(n - 1);
}

// CHECK-LABEL: func.func @recursive
// CHECK: cf.cond_br
// CHECK: return
// CHECK: call @recursive
// CHECK: return
