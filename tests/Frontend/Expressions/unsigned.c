// RUN: clang-to-mlir --disable-opt %s -- | FileCheck %s --enable-var-scope

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

int less_equal(unsigned a, unsigned b) { return a <= b; }

// CHECK-LABEL: func.func @less_equal
// CHECK: arith.cmpi ule

int greater(unsigned a, unsigned b) { return a > b; }

// CHECK-LABEL: func.func @greater
// CHECK: arith.cmpi ugt

int greater_equal(unsigned a, unsigned b) { return a >= b; }

// CHECK-LABEL: func.func @greater_equal
// CHECK: arith.cmpi uge

unsigned mixed_divide(int a, unsigned b) { return a / b; }

// CHECK-LABEL: func.func @mixed_divide
// CHECK: arith.divui

int mixed_compare(int a, unsigned b) { return a < b; }

// CHECK-LABEL: func.func @mixed_compare
// CHECK: arith.cmpi ult

long long wider_signed(long long a, unsigned b) { return a / b; }

// CHECK-LABEL: func.func @wider_signed
// CHECK: %[[B:.*]] = arith.extui %arg1 : i32 to i64
// CHECK: arith.divsi %arg0, %[[B]] : i64

int promoted_divide(unsigned char a, int b) { return a / b; }

// CHECK-LABEL: func.func @promoted_divide
// CHECK: %[[A:.*]] = arith.extui %arg0 : i8 to i32
// CHECK: arith.divsi %[[A]], %arg1 : i32

int signed_shift(int a, unsigned b) { return a >> b; }

// CHECK-LABEL: func.func @signed_shift
// CHECK: arith.shrsi

unsigned mixed_shift(unsigned a, long long count) { return a >> count; }

// CHECK-LABEL: func.func @mixed_shift
// CHECK: %[[COUNT:.*]] = arith.trunci %arg1 : i64 to i32
// CHECK: arith.shrui %arg0, %[[COUNT]] : i32

unsigned long long wide_shift(unsigned long long a, int count) { return a >> count; }

// CHECK-LABEL: func.func @wide_shift
// CHECK: %[[COUNT:.*]] = arith.extsi %arg1 : i32 to i64
// CHECK: arith.shrui %arg0, %[[COUNT]] : i64

int promoted_shift(unsigned char a, unsigned b) { return a >> b; }

// CHECK-LABEL: func.func @promoted_shift
// CHECK: arith.extui {{.*}} : i8 to i32
// CHECK: arith.shrsi

unsigned compound_divide(unsigned a, unsigned b) {
  a /= b;
  return a;
}

// CHECK-LABEL: func.func @compound_divide
// CHECK: arith.divui
// CHECK: memref.store

unsigned compound_modulo(unsigned a, unsigned b) {
  a %= b;
  return a;
}

// CHECK-LABEL: func.func @compound_modulo
// CHECK: arith.remui
// CHECK: memref.store

int mixed_compound(int a, unsigned b) {
  a /= b;
  return a;
}

// CHECK-LABEL: func.func @mixed_compound
// CHECK: arith.divui
// CHECK: memref.store

unsigned char promoted_compound(unsigned char a, int b) {
  a /= b;
  return a;
}

// CHECK-LABEL: func.func @promoted_compound
// CHECK: arith.extui {{.*}} : i8 to i32
// CHECK: %[[R:.*]] = arith.divsi
// CHECK: %[[BYTE:.*]] = arith.trunci %[[R]] : i32 to i8
// CHECK: memref.store %[[BYTE]]

short narrow_mixed_compound(short a, unsigned b) {
  a %= b;
  return a;
}

// CHECK-LABEL: func.func @narrow_mixed_compound
// CHECK: arith.extsi {{.*}} : i16 to i32
// CHECK: %[[R:.*]] = arith.remui
// CHECK: arith.trunci %[[R]] : i32 to i16

unsigned wider_compound(unsigned a, long long b) {
  a /= b;
  return a;
}

// CHECK-LABEL: func.func @wider_compound
// CHECK: arith.extui {{.*}} : i32 to i64
// CHECK: %[[R:.*]] = arith.divsi
// CHECK: arith.trunci %[[R]] : i64 to i32

unsigned compound_wide_count(unsigned a, long long count) {
  a >>= count;
  return a;
}

// CHECK-LABEL: func.func @compound_wide_count
// CHECK: %[[COUNT:.*]] = arith.trunci %arg1 : i64 to i32
// CHECK: arith.shrui {{.*}}, %[[COUNT]] : i32

unsigned char promoted_compound_shift(unsigned char a, unsigned count) {
  a >>= count;
  return a;
}

// CHECK-LABEL: func.func @promoted_compound_shift
// CHECK: arith.extui {{.*}} : i8 to i32
// CHECK: %[[R:.*]] = arith.shrsi
// CHECK: arith.trunci %[[R]] : i32 to i8

int left_shift(int a, unsigned long long count) { return a << count; }

// CHECK-LABEL: func.func @left_shift
// CHECK: %[[COUNT:.*]] = arith.trunci %arg1 : i64 to i32
// CHECK: arith.shli %arg0, %[[COUNT]] : i32

int compound_left_shift(int a, unsigned long long count) {
  a <<= count;
  return a;
}

// CHECK-LABEL: func.func @compound_left_shift
// CHECK: %[[COUNT:.*]] = arith.trunci %arg1 : i64 to i32
// CHECK: arith.shli {{.*}}, %[[COUNT]] : i32
