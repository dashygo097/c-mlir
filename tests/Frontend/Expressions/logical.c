// RUN: clang-to-mlir --disable-opt %s -- | FileCheck %s --enable-var-scope

int logical_and(int flag, int value) { return flag && ++value; }

// CHECK-LABEL: func.func @logical_and
// CHECK: scf.if {{.*}} -> (i1) {
// CHECK: arith.addi
// CHECK: memref.store
// CHECK: scf.yield
// CHECK: } else {
// CHECK: scf.yield
// CHECK: arith.extui {{.*}} : i1 to i32

int logical_or(int flag, int value) { return flag || ++value; }

// CHECK-LABEL: func.func @logical_or
// CHECK: scf.if {{.*}} -> (i1) {
// CHECK: scf.yield
// CHECK: } else {
// CHECK: arith.addi
// CHECK: memref.store
// CHECK: scf.yield

int logical_not(int value) { return !value; }

// CHECK-LABEL: func.func @logical_not
// CHECK: arith.cmpi
// CHECK: arith.xori
// CHECK: arith.extui
