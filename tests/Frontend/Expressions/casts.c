// RUN: clang-to-mlir --disable-opt %s -- | FileCheck %s --enable-var-scope

int signed_widen(signed char a, short b) { return (int)a + (int)b; }

// CHECK-LABEL: func.func @signed_widen
// CHECK: %[[A:.*]] = arith.extsi %arg0 : i8 to i32
// CHECK: %[[B:.*]] = arith.extsi %arg1 : i16 to i32
// CHECK: arith.addi %[[A]], %[[B]] : i32

unsigned unsigned_widen(unsigned char a, unsigned short b) { return (unsigned)a + (unsigned)b; }

// CHECK-LABEL: func.func @unsigned_widen
// CHECK: arith.extui %arg0 : i8 to i32
// CHECK: arith.extui %arg1 : i16 to i32

int narrow(int value) { return (int)(signed char)value + (int)(short)value; }

// CHECK-LABEL: func.func @narrow
// CHECK: arith.trunci %arg0 : i32 to i8
// CHECK: arith.extsi {{.*}} : i8 to i32
// CHECK: arith.trunci %arg0 : i32 to i16
// CHECK: arith.extsi {{.*}} : i16 to i32

float integer_to_float(int a, unsigned b) { return (float)a + (float)b; }

// CHECK-LABEL: func.func @integer_to_float
// CHECK: arith.sitofp %arg0 : i32 to f32
// CHECK: arith.uitofp %arg1 : i32 to f32

int float_to_integer(float a, double b) { return (int)a + (int)b; }

// CHECK-LABEL: func.func @float_to_integer
// CHECK: arith.fptosi %arg0 : f32 to i32
// CHECK: arith.fptosi %arg1 : f64 to i32

unsigned float_to_unsigned(float value) { return (unsigned)value; }

// CHECK-LABEL: func.func @float_to_unsigned
// CHECK: arith.fptoui %arg0 : f32 to i32

float float_narrow(double value) { return (float)value; }

// CHECK-LABEL: func.func @float_narrow
// CHECK: arith.truncf %arg0 : f64 to f32

double float_widen(float value) { return (double)value; }

// CHECK-LABEL: func.func @float_widen
// CHECK: arith.extf %arg0 : f32 to f64
