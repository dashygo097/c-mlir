// RUN: clang-to-mlir --disable-opt %s -- | FileCheck %s --enable-var-scope

int read_vla(int n) {
  int a[n];
  return a[0];
}

// CHECK-LABEL: func.func @read_vla
// CHECK: %[[N64:.*]] = arith.extsi %arg0 : i32 to i64
// CHECK: %[[N:.*]] = arith.index_cast %[[N64]] : i64 to index
// CHECK: %[[A:.*]] = memref.alloca(%[[N]]) : memref<?xi32>
// CHECK: memref.load %[[A]]

int write_vla(int n) {
  int a[n];
  for (int i = 0; i < n; ++i) {
    a[i] = i;
  }
  return a[n - 1];
}

// CHECK-LABEL: func.func @write_vla
// CHECK: %[[A:.*]] = memref.alloca({{.*}}) : memref<?xi32>
// CHECK: scf.for
// CHECK: memref.store {{.*}}, %[[A]]
// CHECK: }
// CHECK: arith.subi %arg0
// CHECK: memref.load %[[A]]
