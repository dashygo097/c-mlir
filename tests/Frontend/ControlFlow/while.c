// RUN: clang-to-mlir --disable-opt %s -- | FileCheck %s --enable-var-scope

int while_loop(int n) {
  int i = 0;
  while (i < n) {
    ++i;
  }
  return i;
}

// CHECK-LABEL: func.func @while_loop
// CHECK: %[[SLOT:.*]] = memref.alloca() : memref<i32>
// CHECK: scf.while : () -> () {
// CHECK: %[[I:.*]] = memref.load %[[SLOT]][]
// CHECK: arith.cmpi slt, %[[I]], %arg0
// CHECK: scf.condition
// CHECK: } do {
// CHECK: arith.addi
// CHECK: memref.store {{.*}}, %[[SLOT]][]
// CHECK: scf.yield
// CHECK: }
// CHECK: %[[R:.*]] = memref.load %[[SLOT]][]
// CHECK: return %[[R]] : i32
