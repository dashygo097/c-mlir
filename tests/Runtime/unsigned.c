// RUN: clang-to-mlir --disable-opt %s -o %t.raw.mlir -- -xc
// RUN: mlir-opt --convert-scf-to-cf --convert-to-llvm %t.raw.mlir -o %t.raw.llvm.mlir
// RUN: mlir-translate --mlir-to-llvmir %t.raw.llvm.mlir -o %t.raw.ll
// RUN: lli %t.raw.ll | FileCheck %s
// RUN: clang-to-mlir %s -o %t.mlir -- -xc
// RUN: mlir-opt --convert-scf-to-cf --convert-to-llvm %t.mlir -o %t.llvm.mlir
// RUN: mlir-translate --mlir-to-llvmir %t.llvm.mlir -o %t.ll
// RUN: lli %t.ll | FileCheck %s

#include <stdio.h>

unsigned divide(unsigned a, unsigned b) { return a / b; }

unsigned modulo(unsigned a, unsigned b) { return a % b; }

int compare(unsigned a, unsigned b) { return (a < b) + 2 * (a <= b) + 4 * (a > b) + 8 * (a >= b); }

unsigned shift(unsigned a, int count) { return a >> count; }

unsigned mixed_divide(int a, unsigned b) { return a / b; }

int mixed_compare(int a, unsigned b) { return a < b; }

long long wider_signed(long long a, unsigned b) { return a / b; }

int promoted_divide(unsigned char a, int b) { return a / b; }

int signed_shift(int a, unsigned count) { return a >> count; }

unsigned mixed_shift(unsigned a, long long count) { return a >> count; }

unsigned long long wide_shift(unsigned long long a, int count) { return a >> count; }

unsigned compound_divide(unsigned a, unsigned b) {
  a /= b;
  return a;
}

unsigned compound_modulo(unsigned a, unsigned b) {
  a %= b;
  return a;
}

unsigned compound_shift(unsigned a, long long count) {
  a >>= count;
  return a;
}

int mixed_compound(int a, unsigned b) {
  a /= b;
  return a;
}

unsigned char promoted_compound(unsigned char a, int b) {
  a /= b;
  return a;
}

short narrow_mixed_compound(short a, unsigned b) {
  a %= b;
  return a;
}

unsigned wider_compound(unsigned a, long long b) {
  a /= b;
  return a;
}

int left_shift(int a, unsigned long long count) { return a << count; }

int compound_left_shift(int a, unsigned long long count) {
  a <<= count;
  return a;
}

int main(void) {
  unsigned high = 4294967294U;
  printf("unsigned %u %u %d %u\n", divide(high, 2), modulo(high, 3), compare(high, 2), shift(high, 1));
  printf("mixed %u %d %lld %d %d\n", mixed_divide(-2, 2U), mixed_compare(-1, 1U), wider_signed(-8, 2U), promoted_divide(200, -2), signed_shift(-8, 1U));
  printf("widths %u %llu %d %d\n", mixed_shift(high, 1LL), wide_shift(9223372036854775808ULL, 1), left_shift(3, 2ULL), compound_left_shift(3, 2ULL));
  printf("compound %u %u %u %d\n", compound_divide(high, 2), compound_modulo(high, 3), compound_shift(high, 1LL), mixed_compound(-2, 2U));
  printf("promotion %u %d %u\n", promoted_compound(200, -2), narrow_mixed_compound(-2, 3U), wider_compound(3, -2LL));
  return 0;
}

// CHECK: unsigned 2147483647 2 12 2147483647
// CHECK-NEXT: mixed 2147483647 0 -4 -100 -4
// CHECK-NEXT: widths 2147483647 4611686018427387904 12 12
// CHECK-NEXT: compound 2147483647 2 2147483647 2147483647
// CHECK-NEXT: promotion 156 2 4294967295
