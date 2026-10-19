// RUN: split-file %s %t
// RUN: clang-to-mlir --disable-opt %t/external.c -- -xc | FileCheck %s --check-prefix=EXTERNAL --implicit-check-not=llvm.call
// RUN: clang-to-mlir --disable-opt %t/math.c -- -xc | FileCheck %s --check-prefix=MATH --implicit-check-not=llvm.call
// RUN: clang-to-mlir --disable-opt %t/same-name.c -- -xc -fno-builtin | FileCheck %s --check-prefix=SAME --implicit-check-not=math.sqrt
// RUN: clang-to-mlir --disable-opt %t/builtin.cpp -- -std=c++17 | FileCheck %s --check-prefix=BUILTIN --implicit-check-not=llvm.call
// RUN: not clang-to-mlir --disable-opt %t/variadic.c -- -xc 2>&1 | FileCheck %s --check-prefix=VARIADIC
// RUN: not clang-to-mlir --disable-opt %t/aggregate.c -- -xc 2>&1 | FileCheck %s --check-prefix=AGGREGATE

//--- external.c
#include <math.h>
#include <stdlib.h>

float external_math(float x, int exponent) {
  return fmodf(x, 2.0f) + ldexpf(x, exponent);
}

void release(void *pointer) { free(pointer); }

// EXTERNAL-DAG: func.func private @fmodf(f32, f32) -> f32
// EXTERNAL-DAG: func.func private @ldexpf(f32, i32) -> f32
// EXTERNAL-DAG: func.func private @free(!llvm.ptr)
// EXTERNAL-LABEL: func.func @external_math
// EXTERNAL: call @fmodf({{.*}}) : (f32, f32) -> f32
// EXTERNAL: call @ldexpf({{.*}}) : (f32, i32) -> f32
// EXTERNAL-LABEL: func.func @release
// EXTERNAL: call @free({{.*}}) : (!llvm.ptr) -> ()

//--- math.c
extern float sqrtf(float);
extern long labs(long);
// Check C int results rather than exposing the math dialect's i1 result.
extern int isfinite(double);

float square_root(float x) { return sqrtf(x); }

long absolute(long x) { return labs(x); }

int finite(double x) { return isfinite(x); }

int classify(float x) { return __builtin_isfinite(x); }

// MATH-LABEL: func.func @square_root
// MATH: math.sqrt {{.*}} : f32
// MATH-LABEL: func.func @absolute
// MATH: math.absi {{.*}} : i64
// MATH-LABEL: func.func @finite
// MATH: %[[FINITE:.*]] = math.isfinite {{.*}} : f64
// MATH: arith.extui %[[FINITE]] : i1 to i32
// MATH-LABEL: func.func @classify
// MATH: %[[GENERIC:.*]] = math.isfinite {{.*}} : f32
// MATH: arith.extui %[[GENERIC]] : i1 to i32

//--- same-name.c
// Names alone do not determine an intrinsic's signature or arity.
extern float sqrtf(float, float);

float unusual(float x) { return sqrtf(x, x); }

// SAME: func.func private @sqrtf(f32, f32) -> f32
// SAME-LABEL: func.func @unusual
// SAME: call @sqrtf({{.*}}) : (f32, f32) -> f32

//--- builtin.cpp
int classify(double x) { return __builtin_isfinite(x); }

// BUILTIN-LABEL: func.func @classify
// BUILTIN: %[[CPP:.*]] = math.isfinite {{.*}} : f64
// BUILTIN: arith.extui %[[CPP]] : i1 to i32

//--- variadic.c
int variadic(int first, ...) { return first; }

// VARIADIC: variadic function definitions are unsupported

//--- aggregate.c
struct Pair {
  int first;
  int second;
};

extern int consume(struct Pair);

int aggregate(struct Pair value) { return consume(value); }

// AGGREGATE: unsupported function call signature
