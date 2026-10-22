// RUN: clang-to-mlir %s -o %t.mlir -- -xc
// RUN: mlir-opt --lower-affine --convert-vector-to-llvm --convert-scf-to-cf --convert-to-llvm --reconcile-unrealized-casts %t.mlir -o %t.llvm.mlir
// RUN: mlir-translate --mlir-to-llvmir %t.llvm.mlir -o %t.ll
// RUN: lli %t.ll | FileCheck %s
// RUN: clang-to-mlir %s --disable-opt -o %t.raw.mlir -- -xc
// RUN: mlir-opt --expand-strided-metadata --convert-scf-to-cf --convert-to-llvm %t.raw.mlir -o %t.raw.llvm.mlir
// RUN: mlir-translate --mlir-to-llvmir %t.raw.llvm.mlir -o %t.raw.ll
// RUN: lli %t.raw.ll | FileCheck %s
// RUN: FileCheck %s --check-prefix=POLICY --input-file=%t.mlir
// RUN: clang-to-mlir %s --raise-scf-to-affine -o %t.affine.mlir -- -xc
// RUN: mlir-opt --lower-affine --convert-vector-to-llvm --convert-scf-to-cf --convert-to-llvm --reconcile-unrealized-casts %t.affine.mlir -o %t.affine.llvm.mlir
// RUN: mlir-translate --mlir-to-llvmir %t.affine.llvm.mlir -o %t.affine.ll
// RUN: lli %t.affine.ll | FileCheck %s

#include <stdio.h>

void add4(float *__restrict out, const float *a, const float *b, int n) {
#pragma cmlir loop vectorize(enable) vectorize_width(4)
  for (int i = 0; i < n; ++i) {
    out[i] = a[i] + b[i];
  }
}

void add3(float *__restrict out, const float *a, const float *b, int n) {
#pragma cmlir loop vectorize(enable) vectorize_width(3)
  for (int i = 3; i < n; ++i) {
    out[i] = a[i] + b[i];
  }
}

void shift(float *out, const float *in, int n) {
  float *next = out + 1;
#pragma cmlir loop vectorize(enable) vectorize_width(4)
  for (int i = 0; i < n; ++i) {
    next[i] = in[i] + 1;
  }
}

float strict_sum(const float *a, int n) {
  float result = 0;
#pragma cmlir loop vectorize(enable) vectorize_width(4)
  for (int i = 0; i < n; ++i) {
    result += a[i];
  }
  return result;
}

// POLICY-LABEL: func.func @shift(
// POLICY-NOT: vector.
// POLICY-LABEL: func.func @strict_sum(
// POLICY-NOT: vector.
// POLICY-LABEL: func.func @sum4(
// POLICY: vector.create_mask
// POLICY: vector.reduction <add>
// POLICY-LABEL: func.func @sum3(
// POLICY: vector.create_mask
// POLICY: vector.reduction <add>
// POLICY-LABEL: func.func @sum_and_xor(
// POLICY: vector.create_mask
// POLICY: vector.reduction <add>
// POLICY: vector.reduction <xor>
// POLICY-LABEL: func.func @check(

unsigned sum4(const unsigned *a, int n, unsigned initial) {
  unsigned result = initial;
#pragma cmlir loop vectorize(enable) vectorize_width(4)
  for (int i = 0; i < n; ++i) {
    result += a[i];
  }
  return result;
}

unsigned sum3(const unsigned *a, int n, unsigned initial) {
  unsigned result = initial;
#pragma cmlir loop vectorize(enable) vectorize_width(3)
  for (int i = 3; i < n; ++i) {
    result += a[i];
  }
  return result;
}

unsigned sum_and_xor(const unsigned *a, int n) {
  unsigned sum = 17;
  unsigned bits = 165;
#pragma cmlir loop vectorize(enable) vectorize_width(4)
  for (int i = 0; i < n; ++i) {
    sum += a[i];
    bits ^= a[i];
  }
  return sum + bits;
}

int check(int n) {
  float a[32], b[32], out[32];
  for (int i = 0; i < 32; ++i) {
    a[i] = i;
    b[i] = 2 * i;
    out[i] = -100;
  }
  add4(out, a, b, n);
  for (int i = 0; i < 32; ++i) {
    float expected = i < n ? 3 * i : -100;
    if (out[i] != expected) {
      return 1;
    }
    out[i] = -100;
  }
  add3(out, a, b, n);
  for (int i = 0; i < 32; ++i) {
    float expected = i >= 3 && i < n ? 3 * i : -100;
    if (out[i] != expected) {
      return 1;
    }
  }
  return 0;
}

int main() {
  int errors = 0;
  // The buffers are deliberately larger than n: a vector tail must respect
  // the loop bound even when the memref still has room for more elements.
  for (int n = -1; n <= 19; ++n) {
    errors += check(n);
  }
  float values[16];
  for (int i = 0; i < 16; ++i) {
    values[i] = 0;
  }
  shift(values, values, 9);
  for (int i = 0; i < 16; ++i) {
    float expected = i < 10 ? i : 0;
    if (values[i] != expected) {
      ++errors;
    }
  }
  float cancellation[4] = {1.0e20f, 1.0f, -1.0e20f, 3.0f};
  if (strict_sum(cancellation, 4) != 3) {
    ++errors;
  }
  unsigned data[23];
  for (int i = 0; i < 23; ++i) {
    data[i] = 0xfffffff0u + i;
  }
  // Cover empty loops, partial vectors, nonzero lower bounds, nonzero seeds,
  // multiple reductions and unsigned wrapping. Extra elements must be masked
  // even when they are inside the backing allocation.
  for (int n = -1; n <= 19; ++n) {
    unsigned expected = 17;
    unsigned expected3 = 23;
    unsigned bits = 165;
    for (int i = 0; i < n; ++i) {
      expected += data[i];
      bits ^= data[i];
      if (i >= 3) {
        expected3 += data[i];
      }
    }
    if (sum4(data, n, 17) != expected || sum3(data, n, 23) != expected3 ||
        sum_and_xor(data, n) != expected + bits) {
      ++errors;
    }
  }
  printf("vector boundaries: %d\n", errors);
  return errors;
}

// CHECK: vector boundaries: 0
