// RUN: not clang-to-mlir --disable-opt %s -- -std=c++17 2>&1 | FileCheck %s

int qualified(volatile int &value) { return value; }

// CHECK: qualified reference bindings are unsupported
