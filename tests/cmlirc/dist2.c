// RUN: cmlirc %s --mem2reg --canonicalize --cse --licm --sscp --symdce --const-prop | FileCheck %s
struct Point {
  float x;
  float y;
};

float dist2(struct Point p1, struct Point p2) {
  float dx = p1.x - p2.x;
  float dy = p1.y - p2.y;
  return dx * dx + dy * dy;
}

// CHECK: func @dist2(%arg0: !llvm.struct<(f32, f32)>, %arg1: !llvm.struct<(f32, f32)>) -> f32
// CHECK: %[[V0:.*]] = llvm.extractvalue %arg0[0] : !llvm.struct<(f32, f32)>
// CHECK: %[[V1:.*]] = llvm.extractvalue %arg1[0] : !llvm.struct<(f32, f32)>
// CHECK: %[[V2:.*]] = arith.subf %[[V0]], %[[V1]] : f32
// CHECK: %[[V3:.*]] = llvm.extractvalue %arg0[1] : !llvm.struct<(f32, f32)>
// CHECK: %[[V4:.*]] = llvm.extractvalue %arg1[1] : !llvm.struct<(f32, f32)>
// CHECK: %[[V5:.*]] = arith.subf %[[V3]], %[[V4]] : f32
// CHECK: %[[V6:.*]] = arith.mulf %[[V2]], %[[V2]] : f32
// CHECK: %[[V7:.*]] = arith.mulf %[[V5]], %[[V5]] : f32
// CHECK: %[[V8:.*]] = arith.addf %[[V6]], %[[V7]] : f32
// CHECK: return %[[V8]] : f32
