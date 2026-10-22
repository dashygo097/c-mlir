// RUN: clang-to-mlir --disable-opt %s -- -ffp-contract=off | FileCheck %s --check-prefix=STRICT --implicit-check-not=fastmath
// RUN: clang-to-mlir --disable-opt %s -- -ffast-math | FileCheck %s --check-prefix=FAST
// RUN: clang-to-mlir --disable-opt %s -- -ffp-contract=fast | FileCheck %s --check-prefix=CONTRACT

float arithmetic(float a, float b, float c) { return (a + b) * c - a / b; }

// STRICT-LABEL: func.func @arithmetic(
// STRICT: arith.addf
// STRICT: arith.mulf
// STRICT: arith.divf
// STRICT: arith.subf
// FAST-LABEL: func.func @arithmetic(
// FAST: arith.addf {{.*}} fastmath<fast> : f32
// FAST: arith.mulf {{.*}} fastmath<fast> : f32
// FAST: arith.divf {{.*}} fastmath<fast> : f32
// FAST: arith.subf {{.*}} fastmath<fast> : f32
// CONTRACT-LABEL: func.func @arithmetic(
// CONTRACT: arith.addf {{.*}} fastmath<contract> : f32
// CONTRACT: arith.mulf {{.*}} fastmath<contract> : f32
// CONTRACT: arith.divf {{.*}} fastmath<contract> : f32
// CONTRACT: arith.subf {{.*}} fastmath<contract> : f32

float compound(float a, float b) {
  a += b;
  a *= b;
  a -= b;
  a /= b;
  return a;
}

// STRICT-LABEL: func.func @compound(
// STRICT: arith.addf
// STRICT: arith.mulf
// STRICT: arith.subf
// STRICT: arith.divf
// FAST-LABEL: func.func @compound(
// FAST: arith.addf {{.*}} fastmath<fast> : f32
// FAST: arith.mulf {{.*}} fastmath<fast> : f32
// FAST: arith.subf {{.*}} fastmath<fast> : f32
// FAST: arith.divf {{.*}} fastmath<fast> : f32
// CONTRACT-LABEL: func.func @compound(
// CONTRACT: arith.addf {{.*}} fastmath<contract> : f32

float unary(float a) {
  ++a;
  --a;
  return -a;
}

// STRICT-LABEL: func.func @unary(
// STRICT: arith.addf
// STRICT: arith.subf
// STRICT: arith.negf
// FAST-LABEL: func.func @unary(
// FAST: arith.addf {{.*}} fastmath<fast> : f32
// FAST: arith.subf {{.*}} fastmath<fast> : f32
// FAST: arith.negf {{.*}} fastmath<fast> : f32
// CONTRACT-LABEL: func.func @unary(
// CONTRACT: arith.negf {{.*}} fastmath<contract> : f32

int compare(float a, float b) { return a < b; }

// STRICT-LABEL: func.func @compare(
// STRICT: arith.cmpf olt
// FAST-LABEL: func.func @compare(
// FAST: arith.cmpf olt, {{.*}} fastmath<fast> : f32
// CONTRACT-LABEL: func.func @compare(
// CONTRACT: arith.cmpf olt, {{.*}} fastmath<contract> : f32

float square_root(float a) { return __builtin_sqrtf(a); }

// STRICT-LABEL: func.func @square_root(
// STRICT: math.sqrt
// FAST-LABEL: func.func @square_root(
// FAST: math.sqrt {{.*}} fastmath<fast> : f32
// CONTRACT-LABEL: func.func @square_root(
// CONTRACT: math.sqrt {{.*}} fastmath<contract> : f32
