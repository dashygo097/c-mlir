// RUN: clang-to-mlir --disable-opt %s -- | FileCheck %s --enable-var-scope

void unroll_count(int *out) {
#pragma cmlir loop unroll(2)
  for (int i = 0; i < 8; ++i) {
    out[i] = i;
  }
}

// CHECK-LABEL: func.func @unroll_count
// CHECK: scf.for
// CHECK: unroll_count = 2 : i32

void no_unroll(int *out) {
#pragma cmlir loop unroll(disable)
  for (int i = 0; i < 8; ++i) {
    out[i] = i;
  }
}

// CHECK-LABEL: func.func @no_unroll
// CHECK: scf.for
// CHECK: nounroll

void vectorize(int *out) {
#pragma cmlir loop vectorize(enable) vectorize_width(4)
  for (int i = 0; i < 8; ++i) {
    out[i] = i;
  }
}

// CHECK-LABEL: func.func @vectorize
// CHECK: scf.for
// CHECK: vectorize
// CHECK-SAME: vectorize_width = 4 : i32
