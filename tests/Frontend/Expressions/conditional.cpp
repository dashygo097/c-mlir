// RUN: clang-to-mlir --disable-opt %s -- -std=c++17 | FileCheck %s --enable-var-scope

int assign(bool condition, int x, int y) {
  (condition ? x : y) = 3;
  return x + y;
}

// CHECK-LABEL: func.func @assign
// CHECK: %[[ADDRESS:.*]] = scf.if {{.*}} -> (!llvm.ptr)
// CHECK: scf.yield {{.*}} : !llvm.ptr
// CHECK: } else {
// CHECK: scf.yield {{.*}} : !llvm.ptr
// CHECK: llvm.store {{.*}}, %[[ADDRESS]] : i32, !llvm.ptr

int increment(bool condition, int x, int y) {
  return ++(condition ? x : y);
}

// CHECK-LABEL: func.func @increment
// CHECK: %[[ADDRESS:.*]] = scf.if {{.*}} -> (!llvm.ptr)
// CHECK: llvm.load %[[ADDRESS]]
// CHECK: arith.addi
// CHECK: llvm.store {{.*}}, %[[ADDRESS]]
