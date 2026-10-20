// RUN: clang-to-mlir %s --math-uplift-to-fma -- -xc | FileCheck %s --check-prefix=FUSED
// RUN: clang-to-mlir --disable-opt %s -- -xc | cmlir-opt --pass-pipeline='builtin.module(default-pipeline{math-uplift-to-fma=true})' | FileCheck %s --check-prefix=FUSED
// RUN: clang-to-mlir %s -- -xc | FileCheck %s --check-prefix=STRICT
// RUN: clang-to-mlir --disable-opt %s -- -xc | cmlir-opt --pass-pipeline='builtin.module(default-pipeline)' | FileCheck %s --check-prefix=STRICT

float mul_add(float a, float b, float c) { return a * b + c; }

float add_mul(float a, float b, float c) { return c + a * b; }

// FUSED-LABEL: func.func @mul_add(
// FUSED: %[[V0:.*]] = math.fma %arg0, %arg1, %arg2 fastmath<contract> : f32
// FUSED: return %[[V0]] : f32
// FUSED-LABEL: func.func @add_mul(
// FUSED: %[[V1:.*]] = math.fma %arg0, %arg1, %arg2 fastmath<contract> : f32
// FUSED: return %[[V1]] : f32

// STRICT-LABEL: func.func @mul_add(
// STRICT: arith.mulf
// STRICT-NEXT: {{.*}}arith.addf
// STRICT-LABEL: func.func @add_mul(
// STRICT: arith.mulf
// STRICT-NEXT: {{.*}}arith.addf
// STRICT-NOT: math.fma
