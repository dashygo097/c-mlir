// RUN: split-file %s %t

// RUN: cmlirc %t/dist2.c | FileCheck %s --check-prefix=CHECK
// RUN: cmlirc %t/dist2_struct2memref.c --struct-to-memref | FileCheck %s --check-prefix=CHECKMEMREF

//--- dist2.c
struct Point {
  float x;
  float y;
};

float dist2(struct Point p1, struct Point p2) {
  float dx = p1.x - p2.x;
  float dy = p1.y - p2.y;
  return dx * dx + dy * dy;
}

// CHECK: func.func @dist2(%arg0: !llvm.struct<(f32, f32)>, %arg1: !llvm.struct<(f32, f32)>) -> f32
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

//--- dist2_struct2memref.c
struct Point {
  float x;
  float y;
};

float dist2(struct Point p1, struct Point p2) {
  float dx = p1.x - p2.x;
  float dy = p1.y - p2.y;
  return dx * dx + dy * dy;
}

// CHECKMEMREF: func.func @dist2(%arg0: memref<?x2xf32>, %arg1: memref<?x2xf32>) -> f32
// CHECKMEMREF-DAG: %[[c0:.*]] = arith.constant 0 : index
// CHECKMEMREF-DAG: %[[c1:.*]] = arith.constant 1 : index
// CHECKMEMREF: %[[V0:.*]] = memref.load %arg0[%c0, %c0] : memref<?x2xf32>
// CHECKMEMREF: %[[V1:.*]] = memref.load %arg1[%c0, %c0] : memref<?x2xf32>
// CHECKMEMREF: %[[V2:.*]] = arith.subf %[[V0]], %[[V1]] : f32
// CHECKMEMREF: %[[V3:.*]] = memref.load %arg0[%c0, %c1] : memref<?x2xf32>
// CHECKMEMREF: %[[V4:.*]] = memref.load %arg1[%c0, %c1] : memref<?x2xf32>
// CHECKMEMREF: %[[V5:.*]] = arith.subf %[[V3]], %[[V4]] : f32
// CHECKMEMREF: %[[V6:.*]] = arith.mulf %[[V2]], %[[V2]] : f32
// CHECKMEMREF: %[[V7:.*]] = arith.mulf %[[V5]], %[[V5]] : f32
// CHECKMEMREF: %[[V8:.*]] = arith.addf %[[V6]], %[[V7]] : f32
// CHECKMEMREF: return %[[V8]] : f32
