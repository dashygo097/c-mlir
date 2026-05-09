// RUN: cmlirc %s -function=sign_extend_casts | FileCheck %s --check-prefix=SIGN_EXTEND
// RUN: cmlirc %s -function=zero_extend_casts | FileCheck %s --check-prefix=ZERO_EXTEND
// RUN: cmlirc %s -function=truncate_casts | FileCheck %s --check-prefix=TRUNCATE
// RUN: cmlirc %s -function=int_float_casts | FileCheck %s --check-prefix=INT_FLOAT
// RUN: cmlirc %s -function=float_int_casts | FileCheck %s --check-prefix=FLOAT_INT

int sign_extend_casts(char c, short s) {
  int a = (int)c;
  int b = (int)s;

  return a + b;
}

// SIGN_EXTEND: func.func @sign_extend_casts(%arg0: i8, %arg1: i16) -> i32
// SIGN_EXTEND: %[[V0:.*]] = arith.extsi %arg0 : i8 to i32
// SIGN_EXTEND: %[[V1:.*]] = arith.extsi %arg1 : i16 to i32
// SIGN_EXTEND: %[[V2:.*]] = arith.addi %[[V0]], %[[V1]] : i32
// SIGN_EXTEND: return %[[V2]] : i32

int zero_extend_casts(unsigned char c, unsigned short s) {
  unsigned int a = (unsigned int)c;
  unsigned int b = (unsigned int)s;

  return a + b;
}

// ZERO_EXTEND: func.func @zero_extend_casts(%arg0: i8, %arg1: i16) -> i32
// ZERO_EXTEND: %[[V0:.*]] = arith.extui %arg0 : i8 to i32
// ZERO_EXTEND: %[[V1:.*]] = arith.extui %arg1 : i16 to i32
// ZERO_EXTEND: %[[V2:.*]] = arith.addi %[[V0]], %[[V1]] : i32
// ZERO_EXTEND: return %[[V2]] : i32

int truncate_casts(int x) {
  char c = (char)x;
  short s = (short)x;

  return (int)c + (int)s;
}

// TRUNCATE: func.func @truncate_casts(%arg0: i32) -> i32
// TRUNCATE: %[[V0:.*]] = arith.trunci %arg0 : i32 to i8
// TRUNCATE: %[[V1:.*]] = arith.trunci %arg0 : i32 to i16
// TRUNCATE: %[[V2:.*]] = arith.extsi %[[V0]] : i8 to i32
// TRUNCATE: %[[V3:.*]] = arith.extsi %[[V1]] : i16 to i32
// TRUNCATE: %[[V4:.*]] = arith.addi %[[V2]], %[[V3]] : i32
// TRUNCATE: return %[[V4]] : i32

float int_float_casts(int x, unsigned int y) {
  float a = (float)x;
  float b = (float)y;

  return a + b;
}

// INT_FLOAT: func.func @int_float_casts(%arg0: i32, %arg1: i32) -> f32
// INT_FLOAT: %[[V0:.*]] = arith.sitofp %arg0 : i32 to f32
// INT_FLOAT: %[[V1:.*]] = arith.uitofp %arg1 : i32 to f32
// INT_FLOAT: %[[V2:.*]] = arith.addf %[[V0]], %[[V1]] : f32
// INT_FLOAT: return %[[V2]] : f32

int float_int_casts(float x, double y) {
  int a = (int)x;
  int b = (int)y;

  return a + b;
}

// FLOAT_INT: func.func @float_int_casts(%arg0: f32, %arg1: f64) -> i32
// FLOAT_INT: %[[V0:.*]] = arith.fptosi %arg0 : f32 to i32
// FLOAT_INT: %[[V1:.*]] = arith.fptosi %arg1 : f64 to i32
// FLOAT_INT: %[[V2:.*]] = arith.addi %[[V0]], %[[V1]] : i32
// FLOAT_INT: return %[[V2]] : i32
