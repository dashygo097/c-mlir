// RUN: cmlir-opt %s --pass-pipeline='builtin.module(default-pipeline)' | FileCheck %s --check-prefix=STRICT
// RUN: cmlir-opt %s --pass-pipeline='builtin.module(default-pipeline{math-uplift-to-fma=true})' | FileCheck %s --check-prefix=FUSED

// STRICT-LABEL: func.func @multiply_add
// STRICT: arith.mulf %arg0, %arg1 : f32
// STRICT: arith.addf {{.*}}, %arg2 : f32
// STRICT-NOT: math.fma
// STRICT: return
// FUSED-LABEL: func.func @multiply_add
// FUSED: math.fma %arg0, %arg1, %arg2 fastmath<contract> : f32
// FUSED-NOT: reassoc
// FUSED: return
func.func @multiply_add(%a: f32, %b: f32, %c: f32) -> f32 {
  %mul = arith.mulf %a, %b : f32
  %sum = arith.addf %mul, %c : f32
  return %sum : f32
}
