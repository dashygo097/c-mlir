// RUN: clang-to-mlir --disable-opt %s -- --target=x86_64-unknown-linux-gnu | FileCheck %s --enable-var-scope

unsigned __int128 invert(unsigned __int128 value) { return ~value; }

// CHECK-LABEL: func.func @invert(%arg0: i128) -> i128
// CHECK: %[[MASK:.*]] = arith.constant -1 : i128
// CHECK: arith.xori %arg0, %[[MASK]] : i128

long double precise(void) { return 0x1.000000000000001p0L; }

// CHECK-LABEL: func.func @precise() -> f80
// CHECK: arith.constant 1.00000000000000000087 : f80

unsigned long long high_bit(void) { return 0x8000000000000000ULL; }

// CHECK-LABEL: func.func @high_bit() -> i64
// CHECK: arith.constant -9223372036854775808 : i64
