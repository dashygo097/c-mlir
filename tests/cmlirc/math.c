// RUN: cmlirc %s -function=fabs_test | FileCheck %s --check-prefix=FABS
// RUN: cmlirc %s -function=sqrt_test | FileCheck %s --check-prefix=SQRT
// RUN: cmlirc %s -function=sin_cos_test | FileCheck %s --check-prefix=SIN_COS
// RUN: cmlirc %s -function=exp_log_test | FileCheck %s --check-prefix=EXP_LOG
// RUN: cmlirc %s -function=pow_test | FileCheck %s --check-prefix=POW
// RUN: cmlirc %s -function=floor_ceil_test | FileCheck %s --check-prefix=FLOOR_CEIL
// RUN: cmlirc %s -function=trunc_round_test | FileCheck %s --check-prefix=TRUNC_ROUND

#include <math.h>

float fabs_test(float x) {
  return fabsf(x);
}

// FABS: func.func @fabs_test(%arg0: f32) -> f32
// FABS: %[[V0:.*]] = math.absf %arg0 : f32
// FABS: return %[[V0]] : f32

float sqrt_test(float x) {
  return sqrtf(x);
}

// SQRT: func.func @sqrt_test(%arg0: f32) -> f32
// SQRT: %[[V0:.*]] = math.sqrt %arg0 : f32
// SQRT: return %[[V0]] : f32

float sin_cos_test(float x) {
  float s = sinf(x);
  float c = cosf(x);

  return s + c;
}

// SIN_COS: func.func @sin_cos_test(%arg0: f32) -> f32
// SIN_COS: %[[V0:.*]] = math.sin %arg0 : f32
// SIN_COS: %[[V1:.*]] = math.cos %arg0 : f32
// SIN_COS: %[[V2:.*]] = arith.addf %[[V0]], %[[V1]] : f32
// SIN_COS: return %[[V2]] : f32

float exp_log_test(float x) {
  float e = expf(x);
  float l = logf(x);

  return e + l;
}

// EXP_LOG: func.func @exp_log_test(%arg0: f32) -> f32
// EXP_LOG: %[[V0:.*]] = math.exp %arg0 : f32
// EXP_LOG: %[[V1:.*]] = math.log %arg0 : f32
// EXP_LOG: %[[V2:.*]] = arith.addf %[[V0]], %[[V1]] : f32
// EXP_LOG: return %[[V2]] : f32

float pow_test(float x, float y) {
  return powf(x, y);
}

// POW: func.func @pow_test(%arg0: f32, %arg1: f32) -> f32
// POW: %[[V0:.*]] = math.powf %arg0, %arg1 : f32
// POW: return %[[V0]] : f32

float floor_ceil_test(float x) {
  float a = floorf(x);
  float b = ceilf(x);

  return a + b;
}

// FLOOR_CEIL: func.func @floor_ceil_test(%arg0: f32) -> f32
// FLOOR_CEIL: %[[V0:.*]] = math.floor %arg0 : f32
// FLOOR_CEIL: %[[V1:.*]] = math.ceil %arg0 : f32
// FLOOR_CEIL: %[[V2:.*]] = arith.addf %[[V0]], %[[V1]] : f32
// FLOOR_CEIL: return %[[V2]] : f32

float trunc_round_test(float x) {
  float a = truncf(x);
  float b = roundf(x);

  return a + b;
}

// TRUNC_ROUND: func.func @trunc_round_test(%arg0: f32) -> f32
// TRUNC_ROUND: %[[V0:.*]] = math.trunc %arg0 : f32
// TRUNC_ROUND: %[[V1:.*]] = math.round %arg0 : f32
// TRUNC_ROUND: %[[V2:.*]] = arith.addf %[[V0]], %[[V1]] : f32
// TRUNC_ROUND: return %[[V2]] : f32
