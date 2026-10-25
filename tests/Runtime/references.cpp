// RUN: clang -std=c++17 -S -emit-llvm %s -o %t.native.ll
// RUN: lli %t.native.ll
// RUN: clang-to-mlir --disable-opt %s -o %t.raw.mlir -- -std=c++17
// RUN: mlir-opt --expand-strided-metadata --lower-affine --convert-scf-to-cf --convert-to-llvm --reconcile-unrealized-casts %t.raw.mlir -o %t.raw.llvm.mlir
// RUN: mlir-translate --mlir-to-llvmir %t.raw.llvm.mlir -o %t.raw.ll
// RUN: lli %t.raw.ll
// RUN: clang-to-mlir %s -o %t.mlir -- -std=c++17
// RUN: mlir-opt --lower-affine --convert-scf-to-cf --convert-to-llvm --reconcile-unrealized-casts %t.mlir -o %t.llvm.mlir
// RUN: mlir-translate --mlir-to-llvmir %t.llvm.mlir -o %t.ll
// RUN: lli %t.ll

struct Pair {
  int first, second;

  Pair &self() { return *this; }
};

void set(int &value, int replacement) { value = replacement; }

int read(const int &value = 6) { return value; }

int &identity(int &value) { return value; }

int &field(Pair &value) { return value.second; }

int sum(const Pair &value) { return value.first + value.second; }

int &&forward(int &&value) { return static_cast<int &&>(value); }

float &floating(float &value) { return value; }

using Array = int[3];

Array &array_ref(Array &values) { return values; }

Array &row(int (&values)[2][3], int index) { return values[index]; }

int &next(int *values, int &index) { return values[index++]; }

int alias(int &first, int &second) {
  first = 4;
  second += first;
  return first;
}

int local_binding(int value) {
  int result = 0;
  for (int i = 0; i < 3; ++i) {
    result += value;
    int &reference = value;
    ++reference;
  }
  return result;
}

int rebind(int *value, int *replacement) {
  int *&reference = value;
  reference = replacement;
  return *value;
}

int main() {
  int value = 1;
  int &reference = value;
  int &chain = reference;
  set(chain, 3);
  if (read(value) != 3 || alias(value, value) != 8) {
    return 1;
  }
  identity(value) = 9;
  set(identity(value), 10);
  if (read(identity(value)) != 10 || read() != 6) {
    return 2;
  }
  Pair pair = {2, 3};
  field(pair.self()) = 7;
  Pair copy = pair.self();
  copy.first = 20;
  if (sum(pair) != 9 || copy.first != 20 || sum(Pair{4, 5}) != 9) {
    return 3;
  }
  const long long &converted = value;
  value = 11;
  int &&rvalue = forward(static_cast<int &&>(value));
  ++rvalue;
  float fp = 1.5f;
  floating(fp) += 2.0f;
  if (converted != 10 || value != 12 || fp != 3.5f || local_binding(2) != 9) {
    return 4;
  }
  int values[3] = {1, 2, 3};
  set(values[1], 6);
  int index = 0;
  next(values, index) = 5;
  int loaded = next(values, index);
  if (index != 2 || loaded != 6 || values[0] != 5 ||
      rebind(&value, &values[1]) != 6 || value != 12) {
    return 5;
  }
  array_ref(values)[2] = 8;
  int matrix[2][3] = {{1, 2, 3}, {4, 5, 6}};
  Array &second = row(matrix, 1);
  array_ref(second)[1] = 9;
  Array &first = matrix[0];
  first[2] = 7;
  if (values[2] != 8 || matrix[1][1] != 9 || matrix[0][2] != 7) {
    return 6;
  }
  return 0;
}
