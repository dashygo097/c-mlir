// RUN: not clang-to-mlir --disable-opt %s -- -std=c++17 2>&1 | FileCheck %s

struct Virtual {
  virtual int value() { return 1; }
};

// CHECK: virtual methods are unsupported
