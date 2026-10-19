// RUN: clang-to-mlir --disable-opt %s -- | FileCheck %s --enable-var-scope

void for_break(int n, int *out) {
  for (int i = 0; i < n; ++i) {
    if (i == 2) {
      break;
    }
    out[i] = i;
  }
}

// CHECK-LABEL: func.func @for_break
// CHECK: cf.cond_br {{.*}}, ^{{bb[0-9]+}}, ^[[EXIT:bb[0-9]+]]
// CHECK: cf.br ^[[EXIT]]

void while_break(int n, int *out) {
  int i = 0;
  while (i < n) {
    if (i == 2) {
      break;
    }
    out[i] = i;
    ++i;
  }
}

// CHECK-LABEL: func.func @while_break
// CHECK: cf.cond_br {{.*}}, ^{{bb[0-9]+}}, ^[[EXIT:bb[0-9]+]]
// CHECK: cf.br ^[[EXIT]]

void do_break(int n, int *out) {
  int i = 0;
  do {
    if (i == 2) {
      break;
    }
    out[i] = i;
    ++i;
  } while (i < n);
}

// CHECK-LABEL: func.func @do_break
// CHECK: cf.cond_br {{.*}}, ^{{bb[0-9]+}}, ^[[EXIT:bb[0-9]+]]
// CHECK: cf.br ^[[EXIT]]

int nested_break(int n) {
  int count = 0;
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      if (j == 2) {
        break;
      }
      ++count;
    }
  }
  return count;
}

// CHECK-LABEL: func.func @nested_break
// CHECK: cf.cond_br {{.*}}, ^{{bb[0-9]+}}, ^[[OUTER_EXIT:bb[0-9]+]]
// CHECK: cf.cond_br {{.*}}, ^{{bb[0-9]+}}, ^[[INNER_EXIT:bb[0-9]+]]
// CHECK: cf.br ^[[INNER_EXIT]]
