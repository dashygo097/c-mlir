// RUN: clang-to-mlir --disable-opt %s -- | FileCheck %s --enable-var-scope

struct Pair {
  float x, y;
};

float field(struct Pair value) { return value.x; }

// CHECK-LABEL: func.func @field(%arg0: !llvm.struct<(f32, f32)>) -> f32
// CHECK: %[[X:.*]] = llvm.extractvalue %arg0[0]
// CHECK: return %[[X]] : f32

float fields(struct Pair a, struct Pair b) { return a.x * b.x + a.y * b.y; }

// CHECK-LABEL: func.func @fields
// CHECK: llvm.extractvalue %arg0[0]
// CHECK: llvm.extractvalue %arg1[0]
// CHECK: arith.mulf
// CHECK: llvm.extractvalue %arg0[1]
// CHECK: llvm.extractvalue %arg1[1]
// CHECK: arith.mulf
// CHECK: arith.addf
