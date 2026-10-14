// RUN: clang-to-mlir %s -function=reduce_sum_nonzero | FileCheck %s

float reduce_sum_nonzero(float *a, int n) {
  float result = 0.0f;
  for (int i = 0; i < n; i++) {
    if (a[i] == 0.0f) {
      continue;
    }
    result += a[i];
  }
  return result;
}

// CHECK: func.func @reduce_sum_nonzero(%arg0: memref<?xf32>, %arg1: i32) -> f32
// CHECK-DAG: %[[cst:.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG: %[[c0_i32:.+]] = arith.constant 0 : i32
// CHECK-DAG: %[[c1_i32:.+]] = arith.constant 1 : i32
// CHECK: cf.br ^[[HEADER:bb[0-9]+]](%[[cst]], %[[c0_i32]] : f32, i32)
// CHECK: ^[[HEADER]](%[[SUM:.*]]: f32, %[[IV:.*]]: i32):
// CHECK: %[[COND:.*]] = arith.cmpi slt, %[[IV]], %arg1 : i32
// CHECK: cf.cond_br %[[COND]], ^[[BODY:bb[0-9]+]], ^[[EXIT:bb[0-9]+]]
// CHECK: ^[[BODY]]:
// CHECK: %[[INDEX:.*]] = arith.index_cast %[[IV]] : i32 to index
// CHECK: %[[ELEMENT:.*]] = memref.load %arg0[%[[INDEX]]] : memref<?xf32>
// CHECK: %[[SKIP:.*]] = arith.cmpf oeq, %[[ELEMENT]], %[[cst]] : f32
// CHECK: cf.cond_br %[[SKIP]], ^[[LATCH:bb[0-9]+]](%[[SUM]] : f32), ^[[ADD:bb[0-9]+]]
// CHECK: ^[[LATCH]](%[[NEXT_SUM:.*]]: f32):
// CHECK: %[[NEXT_IV:.*]] = arith.addi %[[IV]], %[[c1_i32]] : i32
// CHECK: cf.br ^[[HEADER]](%[[NEXT_SUM]], %[[NEXT_IV]] : f32, i32)
// CHECK: ^[[EXIT]]:
// CHECK: return %[[SUM]] : f32
// CHECK: ^[[ADD]]:
// CHECK: %[[VALUE:.*]] = memref.load %arg0[%[[INDEX]]] : memref<?xf32>
// CHECK: %[[ADDED:.*]] = arith.addf %[[SUM]], %[[VALUE]] : f32
// CHECK: cf.br ^[[LATCH]](%[[ADDED]] : f32)
