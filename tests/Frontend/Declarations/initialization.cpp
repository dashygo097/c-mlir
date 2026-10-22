// RUN: clang-to-mlir --disable-opt %s -- -std=c++17 | FileCheck %s --enable-var-scope

class Object {
public:
  int value;
};

int class_object(void) {
  Object object;
  object.value = 3;
  return object.value;
}

// CHECK-LABEL: func.func @class_object()
// CHECK: llvm.alloca {{.*}} x !llvm.struct<(i32)>
// CHECK-NOT: llvm.mlir.undef
// CHECK: llvm.store

struct Pair {
  int first, second;
};

int partial(void) {
  Pair pair = {1};
  return pair.second;
}

// CHECK-LABEL: func.func @partial()
// CHECK: %[[ZERO:.*]] = arith.constant 0 : i32
// CHECK: llvm.store %[[ZERO]]

struct Nested {
  Pair pair;
  int values[2];
};

int nested(void) {
  Nested object = {{3}, {4}};
  return object.pair.second + object.values[1];
}

// CHECK-LABEL: func.func @nested()
// CHECK-COUNT-4: llvm.store

int record_array(Pair (*objects)[2]) { return (*objects)[1].second; }

// CHECK-LABEL: func.func @record_array(%arg0: !llvm.ptr) -> i32
// CHECK: llvm.getelementptr

int scalar_list(void) {
  int value = {};
  return value;
}

// CHECK-LABEL: func.func @scalar_list()
// CHECK: arith.constant 0 : i32
// CHECK: memref.store
