// RUN: split-file %s %t
// RUN: not clang-to-mlir --disable-opt %t/reference.cpp -- -std=c++17 2>&1 | FileCheck %s --check-prefix=REFERENCE
// RUN: not clang-to-mlir --disable-opt %t/global.c -- 2>&1 | FileCheck %s --check-prefix=GLOBAL
// RUN: not clang-to-mlir --disable-opt %t/extern.c -- 2>&1 | FileCheck %s --check-prefix=GLOBAL

//--- reference.cpp
int reference(void) {
  int value = 3;
  int &alias = value;
  return alias;
}

// REFERENCE: reference bindings are unsupported

//--- global.c
int value = 7;

// GLOBAL: global storage is unsupported

//--- extern.c
int read(void) {
  extern int value;
  return value;
}
