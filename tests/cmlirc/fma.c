// RUN: cmlirc %s --fma | FileCheck %s

float fma(float a, float b, float c) {
  return a * b + c;
}

// CHECK: func.func @fma(%arg0: f32, %arg1: f32, %arg2: f32) -> f32
// CHECK: %[[V0:.*]] = math.fma %arg0, %arg1, %arg2 : f32
// CHECK: return %[[V0]] : f32
