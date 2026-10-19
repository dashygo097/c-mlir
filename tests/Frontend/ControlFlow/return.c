// RUN: clang-to-mlir --disable-opt %s -- | FileCheck %s --enable-var-scope

void early_void(int flag) {
  if (flag) {
    return;
  }
  return;
}

// CHECK-LABEL: func.func @early_void
// CHECK: cf.cond_br
// CHECK: return
// CHECK: return

int loop_return(int n) {
  while (n > 0) {
    if (n == 2) {
      return n;
    }
    --n;
  }
  return -1;
}

// CHECK-LABEL: func.func @loop_return
// CHECK: cf.cond_br
// CHECK: cf.cond_br
// CHECK: return
// CHECK: return
