// RUN: split-file %s %t
// RUN: not clang-to-mlir --disable-opt %t/global.c -- 2>&1 | FileCheck %s --check-prefix=GLOBAL
// RUN: not clang-to-mlir --disable-opt %t/extern.c -- 2>&1 | FileCheck %s --check-prefix=GLOBAL

//--- global.c
int value = 7;

// GLOBAL: global storage is unsupported

//--- extern.c
int read(void) {
  extern int value;
  return value;
}
