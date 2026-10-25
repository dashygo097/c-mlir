// RUN: not clang-to-mlir --disable-opt %s -- -std=c++17 2>&1 | FileCheck %s

void native(int *&pointer);

void descriptor_to_native(int *&pointer) { native(pointer); }

// CHECK: native pointer arguments cannot address memref descriptor storage
