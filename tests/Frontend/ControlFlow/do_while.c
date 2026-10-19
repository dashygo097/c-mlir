// RUN: clang-to-mlir --disable-opt %s -- | FileCheck %s --enable-var-scope

int do_while_loop(int n) {
  int i = 0;
  do {
    ++i;
  } while (i < n);
  return i;
}

// CHECK-LABEL: func.func @do_while_loop
// CHECK: %[[SLOT:.*]] = memref.alloca() : memref<i32>
// CHECK: scf.while : () -> () {
// CHECK: arith.addi
// CHECK: memref.store {{.*}}, %[[SLOT]][]
// CHECK: %[[I:.*]] = memref.load %[[SLOT]][]
// CHECK: arith.cmpi slt, %[[I]], %arg0
// CHECK: scf.condition
// CHECK: } do {
// CHECK: scf.yield
// CHECK: }
// CHECK: %[[R:.*]] = memref.load %[[SLOT]][]
// CHECK: return %[[R]] : i32
