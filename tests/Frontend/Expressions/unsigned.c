// RUN: clang-to-mlir --disable-opt %s -- | FileCheck %s --enable-var-scope
// XFAIL: *

// Known bug: unsigned AST operations currently select signed MLIR operations.
// Keep the required unsigned semantics visible until lowering is fixed.
unsigned divide(unsigned a, unsigned b) { return a / b; }

// CHECK-LABEL: func.func @divide
// CHECK: arith.divui

unsigned modulo(unsigned a, unsigned b) { return a % b; }

// CHECK-LABEL: func.func @modulo
// CHECK: arith.remui

int less(unsigned a, unsigned b) { return a < b; }

// CHECK-LABEL: func.func @less
// CHECK: arith.cmpi ult

unsigned shift(unsigned a, unsigned b) { return a >> b; }

// CHECK-LABEL: func.func @shift
// CHECK: arith.shrui

unsigned compound_shift(unsigned a) {
  a >>= 1;
  return a;
}

// CHECK-LABEL: func.func @compound_shift
// CHECK: arith.shrui
