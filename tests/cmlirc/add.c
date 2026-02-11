// RUN: cmlirc %s -function=add | FileCheck %s

int add(int a, int b) { return a + b; }

// CHECK-LABEL: func @add
// CHECK-NEXT: %[[V0:.*]] = arith.addi
// CHECK: return %[[V0]]
