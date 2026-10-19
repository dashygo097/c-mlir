// RUN: clang-to-mlir --disable-opt %s -- | FileCheck %s --enable-var-scope

#include <math.h>

float absolute(float x) { return fabsf(x); }

// CHECK-LABEL: func.func @absolute
// CHECK: math.absf

float square_root(float x) { return sqrtf(x); }

// CHECK-LABEL: func.func @square_root
// CHECK: math.sqrt

float trigonometry(float x) { return sinf(x) + cosf(x); }

// CHECK-LABEL: func.func @trigonometry
// CHECK: %[[SIN:.*]] = math.sin
// CHECK: %[[COS:.*]] = math.cos
// CHECK: arith.addf %[[SIN]], %[[COS]]

float exponential(float x) { return expf(x) + logf(x); }

// CHECK-LABEL: func.func @exponential
// CHECK: math.exp
// CHECK: math.log
// CHECK: arith.addf

float power(float x, float y) { return powf(x, y); }

// CHECK-LABEL: func.func @power
// CHECK: math.powf

float rounding(float x) { return floorf(x) + ceilf(x) + truncf(x) + roundf(x); }

// CHECK-LABEL: func.func @rounding
// CHECK: math.floor
// CHECK: math.ceil
// CHECK: math.trunc
// CHECK: math.round
