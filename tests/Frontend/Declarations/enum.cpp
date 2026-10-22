// RUN: clang-to-mlir --disable-opt %s -- -std=c++17 | FileCheck %s --enable-var-scope

enum class Byte : unsigned char { Max = 255 };
enum class SignedByte : signed char { Negative = -3 };
enum class Flag : bool { Off,
                         On };
enum class Unsigned64 : unsigned long long { Max = ~0ULL };
enum class Signed128 : __int128 { Negative = -1,
                                  Wide = -((__int128)1 << 100) };
enum class Unsigned128 : unsigned __int128 {
  Wide = (unsigned __int128)1 << 100,
  Next
};

Byte byte(void) { return Byte::Max; }

// CHECK-LABEL: func.func @byte() -> i8
// CHECK: %[[VALUE:.*]] = arith.constant -1 : i8
// CHECK: return %[[VALUE]] : i8

int promote_unsigned(Byte value) { return (int)value; }

// CHECK-LABEL: func.func @promote_unsigned(%arg0: i8) -> i32
// CHECK: %[[VALUE:.*]] = arith.extui %arg0 : i8 to i32
// CHECK: return %[[VALUE]] : i32

int promote_signed(SignedByte value) { return (int)value; }

// CHECK-LABEL: func.func @promote_signed(%arg0: i8) -> i32
// CHECK: %[[VALUE:.*]] = arith.extsi %arg0 : i8 to i32
// CHECK: return %[[VALUE]] : i32

Flag flag(void) { return Flag::On; }

// CHECK-LABEL: func.func @flag() -> i1
// CHECK: %[[VALUE:.*]] = arith.constant true
// CHECK: return %[[VALUE]] : i1

Unsigned64 max_unsigned(void) { return Unsigned64::Max; }

// CHECK-LABEL: func.func @max_unsigned() -> i64
// CHECK: %[[VALUE:.*]] = arith.constant -1 : i64
// CHECK: return %[[VALUE]] : i64

Signed128 negative(void) { return Signed128::Negative; }

// CHECK-LABEL: func.func @negative() -> i128
// CHECK: %[[VALUE:.*]] = arith.constant -1 : i128
// CHECK: return %[[VALUE]] : i128

Signed128 wide_signed(void) { return Signed128::Wide; }

// CHECK-LABEL: func.func @wide_signed() -> i128
// CHECK: %[[VALUE:.*]] = arith.constant -1267650600228229401496703205376 : i128
// CHECK: return %[[VALUE]] : i128

Unsigned128 wide_unsigned(void) { return Unsigned128::Wide; }

// CHECK-LABEL: func.func @wide_unsigned() -> i128
// CHECK: %[[VALUE:.*]] = arith.constant 1267650600228229401496703205376 : i128
// CHECK: return %[[VALUE]] : i128

Unsigned128 next_unsigned(void) { return Unsigned128::Next; }

// CHECK-LABEL: func.func @next_unsigned() -> i128
// CHECK: %[[VALUE:.*]] = arith.constant 1267650600228229401496703205377 : i128
// CHECK: return %[[VALUE]] : i128

bool less(Unsigned64 a, Unsigned64 b) { return a < b; }

// CHECK-LABEL: func.func @less(%arg0: i64, %arg1: i64) -> i1
// CHECK: %[[VALUE:.*]] = arith.cmpi ult, %arg0, %arg1 : i64
// CHECK: return %[[VALUE]] : i1
