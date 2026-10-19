// RUN: clang-to-mlir --disable-opt %s -- | FileCheck %s --enable-var-scope

void for_continue(int n, int *out) {
  for (int i = 0; i < n; ++i) {
    if (i == 2) {
      continue;
    }
    out[i] = i;
  }
}

// CHECK-LABEL: func.func @for_continue
// CHECK: cf.br ^[[HEADER:bb[0-9]+]]
// CHECK: cf.cond_br
// CHECK: cf.cond_br {{.*}}, ^[[SKIP:bb[0-9]+]], ^{{bb[0-9]+}}
// CHECK: ^[[LATCH:bb[0-9]+]]:
// CHECK: arith.addi
// CHECK: cf.br ^[[HEADER]]
// CHECK: ^[[SKIP]]:
// CHECK-NEXT: cf.br ^[[LATCH]]

void while_continue(int n, int *out) {
  int i = 0;
  while (i < n) {
    ++i;
    if (i == 2) {
      continue;
    }
    out[i] = i;
  }
}

// CHECK-LABEL: func.func @while_continue
// CHECK: cf.br ^[[HEADER:bb[0-9]+]]
// CHECK: cf.cond_br
// CHECK: cf.cond_br {{.*}}, ^[[SKIP:bb[0-9]+]], ^{{bb[0-9]+}}
// CHECK: ^[[LATCH:bb[0-9]+]]:
// CHECK-NEXT: cf.br ^[[HEADER]]
// CHECK: ^[[SKIP]]:
// CHECK-NEXT: cf.br ^[[LATCH]]

void do_continue(int n, int *out) {
  int i = 0;
  do {
    ++i;
    if (i == 2) {
      continue;
    }
    out[i] = i;
  } while (i < n);
}

// CHECK-LABEL: func.func @do_continue
// CHECK: cf.br ^[[BODY:bb[0-9]+]]
// CHECK: ^[[CONDITION:bb[0-9]+]]:
// CHECK: arith.cmpi slt
// CHECK: cf.cond_br
// CHECK: ^[[BODY]]:
// CHECK: cf.cond_br {{.*}}, ^[[SKIP:bb[0-9]+]], ^{{bb[0-9]+}}
// CHECK: ^[[LATCH:bb[0-9]+]]:
// CHECK-NEXT: cf.br ^[[CONDITION]]
// CHECK: ^[[SKIP]]:
// CHECK-NEXT: cf.br ^[[LATCH]]

int nested_continue(int n) {
  int count = 0;
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      if (j == 2) {
        continue;
      }
      ++count;
    }
  }
  return count;
}

// CHECK-LABEL: func.func @nested_continue
// CHECK: cf.cond_br
// CHECK: cf.cond_br
// CHECK: cf.cond_br {{.*}}, ^[[SKIP:bb[0-9]+]], ^{{bb[0-9]+}}
// CHECK: ^[[INNER_LATCH:bb[0-9]+]]:
// CHECK: arith.addi
// CHECK: ^[[SKIP]]:
// CHECK-NEXT: cf.br ^[[INNER_LATCH]]
