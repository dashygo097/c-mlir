// RUN: clang-to-mlir --disable-opt %s -- | FileCheck %s --enable-var-scope

struct Pair {
  float x, y;
};

float field(struct Pair value) { return value.x; }

// CHECK-LABEL: func.func @field(%arg0: !llvm.struct<(f32, f32)>) -> f32
// CHECK: %[[FIRST:.*]] = llvm.extractvalue %arg0[0]
// CHECK: llvm.store %[[FIRST]]
// CHECK: %[[SECOND:.*]] = llvm.extractvalue %arg0[1]
// CHECK: llvm.store %[[SECOND]]
// CHECK: %[[ADDRESS:.*]] = llvm.getelementptr {{.*}}[0, 0]
// CHECK: %[[X:.*]] = llvm.load %[[ADDRESS]] : !llvm.ptr -> f32
// CHECK: return %[[X]] : f32

float fields(struct Pair a, struct Pair b) { return a.x * b.x + a.y * b.y; }

// CHECK-LABEL: func.func @fields
// CHECK: llvm.extractvalue %arg1[1]
// CHECK: llvm.store
// CHECK: llvm.getelementptr {{.*}}[0, 0]
// CHECK: llvm.load
// CHECK: llvm.getelementptr {{.*}}[0, 0]
// CHECK: llvm.load
// CHECK: arith.mulf
// CHECK: llvm.getelementptr {{.*}}[0, 1]
// CHECK: llvm.load
// CHECK: llvm.getelementptr {{.*}}[0, 1]
// CHECK: llvm.load
// CHECK: arith.mulf
// CHECK: arith.addf
