// RUN: not clang-to-mlir --disable-opt %s -- -std=c++17 2>&1 | FileCheck %s

void indirect(void (*function)()) { function(); }

// CHECK: indirect calls are unsupported
