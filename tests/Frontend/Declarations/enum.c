// RUN: clang-to-mlir --disable-opt %s -- -std=c17 | FileCheck %s --enable-var-scope

enum Number { Zero,
              Negative = -3,
              Next,
              Computed = (1 << 5) + 7 };
typedef enum Number Number;

enum { Anonymous = 17 };

int zero(void) { return Zero; }

// CHECK-LABEL: func.func @zero()
// CHECK: %[[VALUE:.*]] = arith.constant 0 : i32
// CHECK: return %[[VALUE]] : i32

int negative(void) { return Negative; }

// CHECK-LABEL: func.func @negative()
// CHECK: %[[VALUE:.*]] = arith.constant -3 : i32
// CHECK: return %[[VALUE]] : i32

int next(void) { return Next; }

// CHECK-LABEL: func.func @next()
// CHECK: %[[VALUE:.*]] = arith.constant -2 : i32
// CHECK: return %[[VALUE]] : i32

int computed(void) { return Computed; }

// CHECK-LABEL: func.func @computed()
// CHECK-NEXT: %[[VALUE:.*]] = arith.constant 39 : i32
// CHECK-NEXT: return %[[VALUE]] : i32

int anonymous(void) { return Anonymous; }

// CHECK-LABEL: func.func @anonymous()
// CHECK: %[[VALUE:.*]] = arith.constant 17 : i32
// CHECK: return %[[VALUE]] : i32

Number identity(Number value) { return value; }

// CHECK-LABEL: func.func @identity(%arg0: i32) -> i32
// CHECK: return %arg0 : i32

int less(enum Number a, enum Number b) { return a < b; }

// CHECK-LABEL: func.func @less(%arg0: i32, %arg1: i32)
// CHECK: arith.cmpi slt, %arg0, %arg1 : i32
