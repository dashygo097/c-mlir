// RUN: split-file %s %t
// RUN: cmlirc %t/init_vla0.cpp --canonicalize --cse --symdce --mem2reg --const-prop| FileCheck %s --check-prefix=CHECK0
// RUN: cmlirc %t/init_vla1.cpp --canonicalize --cse --symdce --mem2reg --const-prop| FileCheck %s --check-prefix=CHECK1

//--- init_vla0.cpp
int init_vla(int n) {
  int list[n];
  return list[0];
}

// CHECK0: func.func @init_vla(%arg0: i32) -> i32
// CHECK0-DAG: %[[c0:.*]] = arith.constant 0 : index
// CHECK0: %[[V0:.*]] = arith.index_cast %arg0 : i32 to index
// CHECK0: %[[V1:.*]] = memref.alloca(%[[V0]]) : memref<?xi32>
// CHECK0: %[[V2:.*]] = memref.load %[[V1]][%[[c0]]] : memref<?xi32>
// CHECK0: return %[[V2]] : i32

//--- init_vla1.cpp
int init_vla(int n) {
  int list[n];
  for (int i = 0; i < n; i++) {
    list[i] = i;
  }
  return list[n - 1];
}

// CHECK1: func.func @init_vla(%arg0: i32) -> i32
// CHECK1-DAG: %[[c0:.*]] = arith.constant 0 : index
// CHECK1-DAG: %[[c1_i32:.*]] = arith.constant 1 : i32
// CHECK1-DAG: %[[c1:.*]] = arith.constant 1 : index
// CHECK1: %[[V0:.*]] = arith.index_cast %arg0 : i32 to index
// CHECK1: %[[V1:.*]] = memref.alloca(%[[V0]]) : memref<?xi32>
// CHECK1: scf.for %[[V2:.*]] = %[[c0]] to %[[V0]] step %[[c1]] {
// CHECK1:   %[[V3:.*]] = arith.index_cast %[[V2]] : index to i32
// CHECK1:   memref.store %[[V3]], %[[V1]][%[[V2]]] : memref<?xi32>
// CHECK1: }
// CHECK1: %[[V4:.*]] = arith.subi %arg0, %[[c1_i32]] : i32
// CHECK1: %[[V5:.*]] = arith.index_cast %[[V4]] : i32 to index
// CHECK1: %[[V6:.*]] = memref.load %[[V1]][%[[V5]]] : memref<?xi32>
// CHECK1: return %[[V6]] : i32
