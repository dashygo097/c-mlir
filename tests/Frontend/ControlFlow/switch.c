// RUN: clang-to-mlir --disable-opt %s -- | FileCheck %s --enable-var-scope

int fallthrough(int value) {
  int result = 0;
  switch (value) {
  case -1:
    result = 1;
  case 0:
    result += 2;
    break;
  default:
    result = 3;
  }
  return result;
}

// CHECK-LABEL: func.func @fallthrough
// CHECK: arith.constant -1 : i32
// CHECK: arith.cmpi eq
// CHECK: cf.cond_br {{.*}}, ^[[NEGATIVE:bb[0-9]+]], ^{{bb[0-9]+}}
// CHECK: ^[[NEGATIVE]]:
// CHECK: memref.store
// CHECK: cf.br ^[[ZERO:bb[0-9]+]]
// CHECK: ^[[ZERO]]:
// CHECK: arith.addi
// CHECK: cf.br ^[[EXIT:bb[0-9]+]]
// CHECK: ^[[DEFAULT:bb[0-9]+]]:
// CHECK: memref.store
// CHECK: cf.br ^[[EXIT]]
// CHECK: ^[[EXIT]]:
// CHECK: return
// CHECK: cf.cond_br {{.*}}, ^[[ZERO]], ^[[DEFAULT_DISPATCH:bb[0-9]+]]
// CHECK: ^[[DEFAULT_DISPATCH]]:
// CHECK-NEXT: cf.br ^[[DEFAULT]]

int shared_cases(int value) {
  switch (value) {
  case 1:
  case 2:
    return 7;
  default:
    return 9;
  }
}

// CHECK-LABEL: func.func @shared_cases
// CHECK: cf.cond_br {{.*}}, ^[[ONE:bb[0-9]+]], ^{{bb[0-9]+}}
// CHECK: ^[[ONE]]:
// CHECK-NEXT: cf.br ^[[TWO:bb[0-9]+]]
// CHECK: ^[[TWO]]:
// CHECK: arith.constant 7 : i32
// CHECK: return
// CHECK: arith.constant 9 : i32
// CHECK: return
// CHECK: cf.cond_br {{.*}}, ^[[TWO]], ^{{bb[0-9]+}}

int no_default(int value) {
  int result = 0;
  switch (value) {
  case 1:
    result = 1;
    break;
  }
  return result;
}

// CHECK-LABEL: func.func @no_default
// CHECK: cf.cond_br {{.*}}, ^[[CASE:bb[0-9]+]], ^[[NO_MATCH:bb[0-9]+]]
// CHECK: ^[[CASE]]:
// CHECK: memref.store
// CHECK: cf.br ^[[EXIT:bb[0-9]+]]
// CHECK: ^[[EXIT]]:
// CHECK: return
// CHECK: ^[[NO_MATCH]]:
// CHECK-NEXT: cf.br ^[[EXIT]]

int nested_break(int n) {
  int result = 0;
  for (int i = 0; i < n; ++i) {
    switch (i) {
    case 1:
      break;
    default:
      ++result;
    }
    ++result;
  }
  return result;
}

// CHECK-LABEL: func.func @nested_break
// CHECK: cf.br ^[[HEADER:bb[0-9]+]]
// CHECK: cf.cond_br {{.*}}, ^[[BODY:bb[0-9]+]], ^[[LOOP_EXIT:bb[0-9]+]]
// CHECK: ^[[BODY]]:
// CHECK: cf.cond_br {{.*}}, ^[[CASE:bb[0-9]+]], ^{{bb[0-9]+}}
// CHECK: ^[[LATCH:bb[0-9]+]]:
// CHECK: arith.addi
// CHECK: cf.br ^[[HEADER]]
// CHECK: ^[[LOOP_EXIT]]:
// CHECK: return
// CHECK: ^[[CASE]]:
// CHECK-NEXT: cf.br ^[[SWITCH_EXIT:bb[0-9]+]]
// CHECK: ^[[SWITCH_EXIT]]:
// CHECK: arith.addi
// CHECK: cf.br ^[[LATCH]]
