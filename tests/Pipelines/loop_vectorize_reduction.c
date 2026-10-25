// RUN: clang-to-mlir %s -- | FileCheck %s --check-prefix=STRICT --implicit-check-not=vector.
// RUN: clang-to-mlir --disable-opt %s -- -ffast-math | FileCheck %s --check-prefix=RAW --implicit-check-not=vector.
// RUN: clang-to-mlir %s -- -ffast-math | FileCheck %s --check-prefix=VEC
// RUN: clang-to-mlir --disable-opt %s -- -ffast-math | cmlir-opt --default-pipeline --verify-each | FileCheck %s --check-prefix=VEC

float sum3(const float *a, int n, float initial) {
  float result = initial;
#pragma cmlir loop vectorize(enable) vectorize_width(3)
  for (int i = 3; i < n; ++i) {
    result += a[i];
  }
  return result;
}

// STRICT-LABEL: func.func @sum3(
// STRICT: scf.for
// STRICT: arith.addf {{.*}} : f32
// STRICT: return
// RAW-LABEL: func.func @sum3(
// RAW: arith.addf {{.*}} fastmath<fast> : f32
// VEC-LABEL: func.func @sum3(
// VEC: scf.for
// VEC: vector.create_mask {{.*}} : vector<3xi1>
// VEC: vector.transfer_read {{.*}} : memref<?xf32, strided<[1], offset: ?>>, vector<3xf32>
// VEC: arith.addf {{.*}} fastmath<fast> : vector<3xf32>
// VEC: vector.reduction <add>
// VEC: return
