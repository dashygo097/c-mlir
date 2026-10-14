// RUN: not clang-to-mlir %s -function=unsupported -disable-opt -- -xc 2>&1 | FileCheck %s
int unsupported(void) {
  goto exit;
exit:
  return 0;
}

// CHECK: unsupported.c:3:3: error: unsupported statement
