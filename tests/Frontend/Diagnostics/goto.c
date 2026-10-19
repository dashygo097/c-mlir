// RUN: not clang-to-mlir --disable-opt %s -function=unsupported -- -xc 2>&1 | FileCheck %s
int unsupported(void) {
  goto exit;
exit:
  return 0;
}

// CHECK: goto.c:3:3: error: unsupported statement
