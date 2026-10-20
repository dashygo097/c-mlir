// RUN: cmlir-opt %s --arith-cast-prop | FileCheck %s

// Every i32 value is exactly representable in f64.
// CHECK-LABEL: func.func @signed_exact
// CHECK-NOT: arith.sitofp
// CHECK-NOT: arith.fptosi
// CHECK: return %arg0 : i32
func.func @signed_exact(%x: i32) -> i32 {
  %f = arith.sitofp %x : i32 to f64
  %r = arith.fptosi %f : f64 to i32
  return %r : i32
}

// CHECK-LABEL: func.func @unsigned_exact
// CHECK-NOT: arith.uitofp
// CHECK-NOT: arith.fptoui
// CHECK: return %arg0 : i32
func.func @unsigned_exact(%x: i32) -> i32 {
  %f = arith.uitofp %x : i32 to f64
  %r = arith.fptoui %f : f64 to i32
  return %r : i32
}

// f32 cannot preserve all i32 values.
// CHECK-LABEL: func.func @insufficient_precision
// CHECK: arith.sitofp
// CHECK: arith.fptosi
// CHECK: return
func.func @insufficient_precision(%x: i32) -> i32 {
  %f = arith.sitofp %x : i32 to f32
  %r = arith.fptosi %f : f32 to i32
  return %r : i32
}

// CHECK-LABEL: func.func @unsigned_insufficient_precision
// CHECK: arith.uitofp
// CHECK: arith.fptoui
// CHECK: return
func.func @unsigned_insufficient_precision(%x: i64) -> i64 {
  %f = arith.uitofp %x : i64 to f64
  %r = arith.fptoui %f : f64 to i64
  return %r : i64
}

// Matching bit widths do not justify mixing signed and unsigned conversions.
// CHECK-LABEL: func.func @mixed_signedness
// CHECK: arith.uitofp
// CHECK: arith.fptosi
// CHECK: return
func.func @mixed_signedness(%x: i32) -> i32 {
  %f = arith.uitofp %x : i32 to f64
  %r = arith.fptosi %f : f64 to i32
  return %r : i32
}

// Arithmetic between conversions must retain floating-point semantics.
// CHECK-LABEL: func.func @intervening_arithmetic
// CHECK: arith.sitofp
// CHECK: arith.addf
// CHECK: arith.fptosi
// CHECK: return
func.func @intervening_arithmetic(%x: i32, %delta: f64) -> i32 {
  %f = arith.sitofp %x : i32 to f64
  %sum = arith.addf %f, %delta : f64
  %r = arith.fptosi %sum : f64 to i32
  return %r : i32
}

// f16 precision alone is insufficient: its finite exponent range also matters.
// CHECK-LABEL: func.func @insufficient_exponent
// CHECK: arith.sitofp
// CHECK: arith.fptosi
// CHECK: return
func.func @insufficient_exponent(%x: i32) -> i32 {
  %f = arith.sitofp %x : i32 to f16
  %r = arith.fptosi %f : f16 to i32
  return %r : i32
}
