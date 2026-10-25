// RUN: clang-to-mlir --disable-opt %s -- -std=c++17 | FileCheck %s --enable-var-scope --implicit-check-not=llvm

// CHECK-LABEL: func.func @rebind(
// CHECK-SAME: %arg0: memref<memref<?xi32
// CHECK: memref.store %arg1, %arg0[]
void rebind(int *&pointer, int *replacement) { pointer = replacement; }

// CHECK-LABEL: func.func @identity(
// CHECK-SAME: %arg0: memref<memref<?xi32
// CHECK-NEXT: return %arg0 : memref<memref<?xi32
int *&identity(int *&pointer) { return pointer; }

// CHECK-LABEL: func.func @local_alias
// CHECK: %[[SLOT:.*]] = memref.alloca() : memref<memref<?xi32
// CHECK: memref.store %arg0, %[[SLOT]][]
// CHECK: memref.store %arg1, %[[SLOT]][]
// CHECK: %[[POINTER:.*]] = memref.load %[[SLOT]][]
// CHECK: memref.load %[[POINTER]][
int local_alias(int *pointer, int *replacement) {
  int *&alias = pointer;
  alias = replacement;
  return *pointer;
}

// CHECK-LABEL: func.func @returned_alias
// CHECK: call @identity
// CHECK: memref.store
// CHECK: memref.load
int returned_alias(int *&pointer, int *replacement) {
  identity(pointer) = replacement;
  return *pointer;
}

// CHECK-LABEL: func.func @indirect
// CHECK: memref.alloca() : memref<memref<?xi32
// CHECK: memref.reinterpret_cast
// CHECK: memref.store
// CHECK: return
int indirect(int *first, int *second) {
  int *pointer = first;
  int **address = &pointer;
  *address = second;
  return *pointer;
}

// CHECK-LABEL: func.func @advance(
// CHECK-SAME: %arg0: memref<memref<?xi32
// CHECK: %[[OLD:.*]] = memref.load %arg0[]
// CHECK: memref.extract_strided_metadata %[[OLD]]
// CHECK: %[[ADD:.*]] = memref.reinterpret_cast
// CHECK: memref.store %[[ADD]], %arg0[]
// CHECK: %[[POST:.*]] = memref.load %arg0[]
// CHECK: %[[NEXT:.*]] = memref.reinterpret_cast
// CHECK: memref.store %[[NEXT]], %arg0[]
// CHECK: memref.load %[[POST]][
int advance(int *&pointer, int offset) {
  pointer += offset;
  return *pointer++;
}
