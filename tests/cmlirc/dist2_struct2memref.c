// RUN: cmlirc %s --mem2reg --canonicalize --cse --licm --sscp --symdce --const-prop --struct-to-memref | FileCheck %s
struct Point {
  float x;
  float y;
};

float dist2(struct Point p1, struct Point p2) {
  float dx = p1.x - p2.x;
  float dy = p1.y - p2.y;
  return dx * dx + dy * dy;
}

// CHECK: func @dist2(%arg0: memref<?x2xf32>, %arg1: memref<?x2xf32>) -> f32
// CHECK-DAG: %[[c0:.*]] = arith.constant 0 : index
// CHECK-DAG: %[[c1:.*]] = arith.constant 1 : index
// CHECK: %[[V0:.*]] = memref.load %arg0[%c0, %c0] : memref<?x2xf32>
// CHECK: %[[V1:.*]] = memref.load %arg1[%c0, %c0] : memref<?x2xf32>
// CHECK: %[[V2:.*]] = arith.subf %[[V0]], %[[V1]] : f32
// CHECK: %[[V3:.*]] = memref.load %arg0[%c0, %c1] : memref<?x2xf32>
// CHECK: %[[V4:.*]] = memref.load %arg1[%c0, %c1] : memref<?x2xf32>
// CHECK: %[[V5:.*]] = arith.subf %[[V3]], %[[V4]] : f32
// CHECK: %[[V6:.*]] = arith.mulf %[[V2]], %[[V2]] : f32
// CHECK: %[[V7:.*]] = arith.mulf %[[V5]], %[[V5]] : f32
// CHECK: %[[V8:.*]] = arith.addf %[[V6]], %[[V7]] : f32
// CHECK: return %[[V8]] : f32
