// RUN: cmlir-opt %s --enable-fp-contract | FileCheck %s

// The policy preserves existing flags and does not itself fuse operations.
// CHECK-LABEL: func.func @permit_contraction
// CHECK: arith.mulf {{.*}} fastmath<contract> : f32
// CHECK: arith.addf {{.*}} fastmath<reassoc,contract> : f32
// CHECK-NOT: math.fma
// CHECK: return
func.func @permit_contraction(%a: f32, %b: f32, %c: f32) -> f32 {
  %mul = arith.mulf %a, %b : f32
  %sum = arith.addf %mul, %c fastmath<reassoc> : f32
  return %sum : f32
}

// CHECK-LABEL: func.func @other_operations
// CHECK: arith.subf %arg0, %arg1 : f32
// CHECK: arith.divf {{.*}} : f32
// CHECK: return
func.func @other_operations(%a: f32, %b: f32) -> f32 {
  %sub = arith.subf %a, %b : f32
  %div = arith.divf %sub, %b : f32
  return %div : f32
}
