// RUN: clang-to-mlir --disable-opt %s -- | FileCheck %s --enable-var-scope

int choose(int condition, int a, int b) {
  if (condition) {
    return a;
  } else {
    return b;
  }
}

// CHECK-LABEL: func.func @choose
// CHECK: cf.cond_br {{.*}}, ^[[THEN:bb[0-9]+]], ^[[ELSE:bb[0-9]+]]
// CHECK: ^[[THEN]]:
// CHECK: return %arg1 : i32
// CHECK: ^[[ELSE]]:
// CHECK: return %arg2 : i32

int no_else(int condition) {
  int value = 0;
  if (condition) {
    value = 1;
  }
  return value;
}

// CHECK-LABEL: func.func @no_else
// CHECK: cf.cond_br {{.*}}, ^[[THEN:bb[0-9]+]], ^[[MERGE:bb[0-9]+]]
// CHECK: ^[[THEN]]:
// CHECK: memref.store
// CHECK: cf.br ^[[MERGE]]
// CHECK: ^[[MERGE]]:
// CHECK: memref.load
// CHECK: return

int nested(int a, int b) {
  if (a) {
    if (b) {
      return 1;
    }
    return 2;
  }
  return 3;
}

// CHECK-LABEL: func.func @nested
// CHECK: cf.cond_br
// CHECK: cf.cond_br
// CHECK: return
