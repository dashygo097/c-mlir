// RUN: clang-to-mlir --disable-opt %s -- | FileCheck %s --enable-var-scope

int choose(int condition, int a, int b) { return condition ? a : b; }

// CHECK-LABEL: func.func @choose
// CHECK: %[[R:.*]] = scf.if {{.*}} -> (i32) {
// CHECK: scf.yield %arg1 : i32
// CHECK: } else {
// CHECK: scf.yield %arg2 : i32
// CHECK: return %[[R]] : i32

int side_effect(int condition, int value) { return condition ? ++value : --value; }

// CHECK-LABEL: func.func @side_effect
// CHECK: scf.if
// CHECK: arith.addi
// CHECK: memref.store
// CHECK: scf.yield
// CHECK: } else {
// CHECK: arith.subi
// CHECK: memref.store
// CHECK: scf.yield
