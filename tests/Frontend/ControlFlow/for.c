// RUN: clang-to-mlir --disable-opt %s -- | FileCheck %s --enable-var-scope

int ascending(int n) {
  int sum = 0;
  for (int i = 0; i < n; ++i) {
    sum += i;
  }
  return sum;
}

// CHECK-LABEL: func.func @ascending
// CHECK: %[[SUM:.*]] = memref.alloca() : memref<i32>
// CHECK: scf.for
// CHECK: arith.addi
// CHECK: memref.store {{.*}}, %[[SUM]][]
// CHECK: }
// CHECK: memref.load %[[SUM]][]
// CHECK: return

int descending(int n) {
  int sum = 0;
  for (int i = n; i >= 0; i -= 2) {
    sum += i;
  }
  return sum;
}

// CHECK-LABEL: func.func @descending
// CHECK: scf.for
// CHECK: arith.subi
// CHECK: arith.addi

int nested(int n) {
  int sum = 0;
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      sum += i * j;
    }
  }
  return sum;
}

// CHECK-LABEL: func.func @nested
// CHECK: scf.for
// CHECK: scf.for
// CHECK: arith.muli

int changing_bound(void) {
  int n = 5, sum = 0;
  for (int i = 0; i < n; ++i) {
    sum += i;
    --n;
  }
  return sum;
}

// CHECK-LABEL: func.func @changing_bound
// CHECK-NOT: scf.for
// CHECK: cf.cond_br
// CHECK: arith.subi
// CHECK: cf.br
