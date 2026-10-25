// RUN: clang-to-mlir %s --lift-cf-to-scf --function=max3 -- | FileCheck %s --check-prefix=CHECK3 --implicit-check-not=cf.br --implicit-check-not=cf.cond_br --implicit-check-not=cf.switch
// RUN: clang-to-mlir --disable-opt %s --function=max3 -- | cmlir-opt --pass-pipeline='builtin.module(default-pipeline{lift-cf-to-scf=true})' | FileCheck %s --check-prefix=CHECK3 --implicit-check-not=cf.br --implicit-check-not=cf.cond_br --implicit-check-not=cf.switch
// RUN: clang-to-mlir %s --lift-cf-to-scf --function=fibonacci -- | FileCheck %s --check-prefix=FIB --implicit-check-not=cf.br --implicit-check-not=cf.cond_br --implicit-check-not=cf.switch
// RUN: clang-to-mlir --disable-opt %s --function=fibonacci -- | cmlir-opt --pass-pipeline='builtin.module(default-pipeline{lift-cf-to-scf=true})' | FileCheck %s --check-prefix=FIB --implicit-check-not=cf.br --implicit-check-not=cf.cond_br --implicit-check-not=cf.switch
// RUN: clang-to-mlir %s --lift-cf-to-scf --function=reduce_sum_nonzero -- | FileCheck %s --check-prefix=CONT --implicit-check-not=cf.br --implicit-check-not=cf.cond_br --implicit-check-not=cf.switch
// RUN: clang-to-mlir --disable-opt %s --function=reduce_sum_nonzero -- | cmlir-opt --pass-pipeline='builtin.module(default-pipeline{lift-cf-to-scf=true})' | FileCheck %s --check-prefix=CONT --implicit-check-not=cf.br --implicit-check-not=cf.cond_br --implicit-check-not=cf.switch

int max3(int a, int b, int c) {
  if (a > b && a > c) {
    return a;
  } else if (b > c) {
    return b;
  } else {
    return c;
  }
}

// CHECK3: func.func @max3(%arg0: i32, %arg1: i32, %arg2: i32) -> i32
// CHECK3-DAG: %[[false:.*]] = arith.constant false
// CHECK3: %[[V0:.*]] = arith.cmpi sgt, %arg0, %arg1 : i32
// CHECK3: %[[V2:.*]] = scf.if %[[V0]] -> (i1) {
// CHECK3: %[[V1:.*]] = arith.cmpi sgt, %arg0, %arg2 : i32
// CHECK3: scf.yield %[[V1]] : i1
// CHECK3: } else {
// CHECK3: scf.yield %[[false]] : i1
// CHECK3: }
// CHECK3: %[[RESULT:.*]] = scf.if %[[V2]] -> (i32) {
// CHECK3: scf.yield %arg0 : i32
// CHECK3: } else {
// CHECK3: %[[MAX_BC:.*]] = arith.maxsi %arg1, %arg2 : i32
// CHECK3: scf.yield %[[MAX_BC]] : i32
// CHECK3: }
// CHECK3: return %[[RESULT]] : i32

int fibonacci(int n) {
  if (n <= 1) {
    return n;
  }

  return fibonacci(n - 1) + fibonacci(n - 2);
}

// FIB: func.func @fibonacci(%arg0: i32) -> i32
// FIB-DAG: %[[c1_i32:.*]] = arith.constant 1 : i32
// FIB-DAG: %[[c2_i32:.*]] = arith.constant 2 : i32
// FIB: %[[V0:.*]] = arith.cmpi sle, %arg0, %[[c1_i32]] : i32
// FIB: %[[RESULT:.*]] = scf.if %[[V0]] -> (i32) {
// FIB: scf.yield %arg0 : i32
// FIB: } else {
// FIB: %[[V1:.*]] = arith.subi %arg0, %[[c1_i32]] : i32
// FIB: %[[V2:.*]] = func.call @fibonacci(%[[V1]]) : (i32) -> i32
// FIB: %[[V3:.*]] = arith.subi %arg0, %[[c2_i32]] : i32
// FIB: %[[V4:.*]] = func.call @fibonacci(%[[V3]]) : (i32) -> i32
// FIB: %[[V5:.*]] = arith.addi %[[V2]], %[[V4]] : i32
// FIB: scf.yield %[[V5]] : i32
// FIB: }
// FIB: return %[[RESULT]] : i32

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

// CONT: func.func @reduce_sum_nonzero(%arg0: memref<?xf32, strided<[1], offset: ?>>, %arg1: i32) -> f32
// CONT-DAG: %[[cst:.+]] = arith.constant 0.000000e+00 : f32
// CONT-DAG: %[[c0_i32:.+]] = arith.constant 0 : i32
// CONT-DAG: %[[c1_i32:.+]] = arith.constant 1 : i32
// CONT: %[[LOOP:.*]]:2 = scf.while (%[[SUM:.*]] = %[[cst]], %[[IV:.*]] = %[[c0_i32]]) : (f32, i32) -> (f32, i32) {
// CONT: %[[COND:.*]] = arith.cmpi slt, %[[IV]], %arg1 : i32
// CONT: scf.condition(%[[COND]]) %[[SUM]], %[[IV]] : f32, i32
// CONT: } do {
// CONT: ^bb0(%[[BODY_SUM:.*]]: f32, %[[BODY_IV:.*]]: i32):
// CONT: %[[INDEX:.*]] = arith.index_cast %[[BODY_IV]] : i32 to index
// CONT: %[[ELEMENT:.*]] = memref.load %arg0[%[[INDEX]]] : memref<?xf32, strided<[1], offset: ?>>
// CONT: %[[SKIP:.*]] = arith.cmpf oeq, %[[ELEMENT]], %[[cst]] : f32
// CONT: %[[NEXT_SUM:.*]] = scf.if %[[SKIP]] -> (f32) {
// CONT: scf.yield %[[BODY_SUM]] : f32
// CONT: } else {
// CONT: %[[VALUE:.*]] = memref.load %arg0[%[[INDEX]]] : memref<?xf32, strided<[1], offset: ?>>
// CONT: %[[ADDED:.*]] = arith.addf %[[BODY_SUM]], %[[VALUE]] : f32
// CONT: scf.yield %[[ADDED]] : f32
// CONT: }
// CONT: %[[NEXT_IV:.*]] = arith.addi %[[BODY_IV]], %[[c1_i32]] : i32
// CONT: scf.yield %[[NEXT_SUM]], %[[NEXT_IV]] : f32, i32
// CONT: }
// CONT: return %[[LOOP]]#0 : f32
