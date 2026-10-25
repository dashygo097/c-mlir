// RUN: clang-to-mlir --disable-opt %s -- -std=c++17 | FileCheck %s --enable-var-scope --implicit-check-not=llvm

// CHECK-LABEL: func.func @identity(%arg0: memref<i32, strided<[], offset: ?>>) -> memref<i32, strided<[], offset: ?>>
// CHECK-NEXT: return %arg0 : memref<i32, strided<[], offset: ?>>
int &identity(int &value) { return value; }

// CHECK-LABEL: func.func @set(%arg0: memref<i32, strided<[], offset: ?>>, %arg1: i32)
// CHECK: memref.store %arg1, %arg0[]
void set(int &value, int replacement) { value = replacement; }

// CHECK-LABEL: func.func @forward(%arg0: memref<i32, strided<[], offset: ?>>) -> memref<i32, strided<[], offset: ?>>
// CHECK-NEXT: return %arg0 : memref<i32, strided<[], offset: ?>>
int &&forward(int &&value) { return static_cast<int &&>(value); }

// CHECK-LABEL: func.func @call_result
// CHECK: %[[ADDRESS:.*]] = call @identity(%arg0)
// CHECK: memref.store {{.*}}, %[[ADDRESS]][]
// CHECK: memref.load %arg0[]
int call_result(int &value) {
  identity(value) = 7;
  return value;
}

using Array = int[3];

// CHECK-LABEL: func.func @array_ref(%arg0: memref<3xi32, strided<[1], offset: ?>>) -> memref<3xi32, strided<[1], offset: ?>>
// CHECK-NEXT: return %arg0 : memref<3xi32, strided<[1], offset: ?>>
Array &array_ref(Array &values) { return values; }

// CHECK-LABEL: func.func @element
// CHECK: memref.subview %arg0
// CHECK: return {{.*}} : memref<i32, strided<[], offset: ?>>
int &element(Array &values, int index) { return values[index]; }

// CHECK-LABEL: func.func @reference(%arg0: memref<i32, strided<[], offset: ?>>)
// CHECK: %[[OLD:.*]] = memref.load %arg0[]
// CHECK: %[[NEW:.*]] = arith.addi %[[OLD]],
// CHECK: memref.store %[[NEW]], %arg0[]
// CHECK: return %[[NEW]] : i32
int reference(int &n) { return ++n; }

// CHECK-LABEL: func.func @local_reference()
// CHECK: %[[SLOT:.*]] = memref.alloca() : memref<i32>
// CHECK-NOT: memref.alloca
// CHECK: %[[VALUE:.*]] = memref.load %[[SLOT]][]
// CHECK: return %[[VALUE]] : i32
int local_reference(void) {
  int value = 3;
  int &alias = value;
  return alias;
}
