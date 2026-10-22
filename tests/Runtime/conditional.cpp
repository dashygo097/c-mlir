// RUN: clang-to-mlir --disable-opt %s -o %t.raw.mlir -- -std=c++17
// RUN: mlir-opt --expand-strided-metadata --convert-scf-to-cf --convert-to-llvm --reconcile-unrealized-casts %t.raw.mlir -o %t.raw.llvm.mlir
// RUN: mlir-translate --mlir-to-llvmir %t.raw.llvm.mlir -o %t.raw.ll
// RUN: lli %t.raw.ll
// RUN: clang-to-mlir %s -o %t.mlir -- -std=c++17
// RUN: mlir-opt --convert-scf-to-cf --convert-to-llvm --reconcile-unrealized-casts %t.mlir -o %t.llvm.mlir
// RUN: mlir-translate --mlir-to-llvmir %t.llvm.mlir -o %t.ll
// RUN: lli %t.ll

int assign(bool condition) {
  int x = 1, y = 2;
  (condition ? x : y) = 7;
  return x * 10 + y;
}

int selected_index(bool condition) {
  int values[3] = {};
  int first = 0, second = 1;
  ++(condition ? values[first++] : values[second++]);
  return first * 100 + second * 10 + values[0] + values[1];
}

struct Object {
  int value;
};

int selected_object(bool condition) {
  Object first = {1}, second = {2};
  (condition ? first : second).value = 7;
  return first.value * 10 + second.value;
}

int selected_pointer(bool condition) {
  int first = 1, second = 2;
  int *a = &first, *b = &second;
  return *(condition ? a : b);
}

void set(int *value) { *value = 7; }

int selected_void(bool condition) {
  int first = 1, second = 2;
  condition ? set(&first) : set(&second);
  return first * 10 + second;
}

int main(void) {
  if (assign(true) != 72 || assign(false) != 17) {
    return 1;
  }
  if (selected_index(true) != 111 || selected_index(false) != 21) {
    return 2;
  }
  if (static_cast<int>(3.75) != 3 || (float)7U != 7.0f) {
    return 3;
  }
  if (selected_object(true) != 72 || selected_object(false) != 17 ||
      selected_pointer(true) != 1 || selected_pointer(false) != 2 ||
      selected_void(true) != 72 || selected_void(false) != 17) {
    return 4;
  }
  return 0;
}
