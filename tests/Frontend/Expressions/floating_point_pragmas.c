// RUN: clang-to-mlir --disable-opt %s -- | FileCheck %s --check-prefix=LOCAL
// RUN: clang-to-mlir --disable-opt %s -- -ffast-math | FileCheck %s --check-prefix=FAST

float relaxed(float a, float b) {
#pragma clang fp reassociate(on)
  return a + b;
}

// LOCAL-LABEL: func.func @relaxed(
// LOCAL: arith.addf {{.*}} fastmath<reassoc> : f32
// FAST-LABEL: func.func @relaxed(
// FAST: arith.addf {{.*}} fastmath<fast> : f32

float precise(float a, float b) {
#pragma clang fp reassociate(off)
  return a + b;
}

// LOCAL-LABEL: func.func @precise(
// LOCAL: arith.addf {{.*}} : f32
// FAST-LABEL: func.func @precise(
// FAST-NOT: reassoc
// FAST-NOT: fastmath<fast>
// FAST: return

// The previous function's pragma must not leak into this expression.
float inherited(float a, float b) { return a + b; }

// LOCAL-LABEL: func.func @inherited(
// LOCAL: arith.addf {{.*}} : f32
// FAST-LABEL: func.func @inherited(
// FAST: arith.addf {{.*}} fastmath<fast> : f32
