// RUN: cmlir-opt %s --pass-pipeline='builtin.module(default-pipeline)' | FileCheck %s --check-prefix=STRICT
// RUN: cmlir-opt %s --pass-pipeline='builtin.module(default-pipeline{math-uplift-to-fma=true})' | FileCheck %s --check-prefix=UPLIFT

// STRICT-LABEL: func.func @multiply_add
// STRICT: arith.mulf %arg0, %arg1 : f32
// STRICT: arith.addf {{.*}}, %arg2 : f32
// STRICT-NOT: math.fma
// STRICT: return
// UPLIFT-LABEL: func.func @multiply_add
// UPLIFT: arith.mulf %arg0, %arg1 : f32
// UPLIFT: arith.addf {{.*}}, %arg2 : f32
// UPLIFT-NOT: math.fma
// UPLIFT: return
func.func @multiply_add(%a: f32, %b: f32, %c: f32) -> f32 {
  %mul = arith.mulf %a, %b : f32
  %sum = arith.addf %mul, %c : f32
  return %sum : f32
}

// STRICT-LABEL: func.func @permitted_multiply_add
// STRICT: arith.mulf %arg0, %arg1 fastmath<contract> : f32
// STRICT: arith.addf {{.*}}, %arg2 fastmath<contract> : f32
// STRICT-NOT: math.fma
// STRICT: return
// UPLIFT-LABEL: func.func @permitted_multiply_add
// UPLIFT: math.fma %arg0, %arg1, %arg2 fastmath<contract> : f32
// UPLIFT-NOT: reassoc
// UPLIFT: return
func.func @permitted_multiply_add(%a: f32, %b: f32, %c: f32) -> f32 {
  %mul = arith.mulf %a, %b fastmath<contract> : f32
  %sum = arith.addf %mul, %c fastmath<contract> : f32
  return %sum : f32
}

// Both operations must permit contraction.
// STRICT-LABEL: func.func @partially_permitted
// STRICT: arith.mulf %arg0, %arg1 fastmath<contract> : f32
// STRICT: arith.addf {{.*}}, %arg2 : f32
// STRICT-NOT: math.fma
// STRICT: return
// UPLIFT-LABEL: func.func @partially_permitted
// UPLIFT: arith.mulf %arg0, %arg1 fastmath<contract> : f32
// UPLIFT: arith.addf {{.*}}, %arg2 : f32
// UPLIFT-NOT: math.fma
// UPLIFT: return
func.func @partially_permitted(%a: f32, %b: f32, %c: f32) -> f32 {
  %mul = arith.mulf %a, %b fastmath<contract> : f32
  %sum = arith.addf %mul, %c : f32
  return %sum : f32
}
