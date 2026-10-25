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
// RUN: clang-to-mlir --inline %s -o %t.inline.mlir -- -std=c++17
// RUN: mlir-opt --lower-affine --convert-scf-to-cf --convert-to-llvm --reconcile-unrealized-casts %t.inline.mlir -o %t.inline.llvm.mlir
// RUN: mlir-translate --mlir-to-llvmir %t.inline.llvm.mlir -o %t.inline.ll
// RUN: lli %t.inline.ll

int steps(int *&pointer, int offset) {
  pointer += offset;
  int post = *pointer++;
  int pre = *++pointer;
  int post_back = *pointer--;
  int pre_back = *--pointer;
  pointer -= offset;
  return post * 1000 + pre * 100 + post_back * 10 + pre_back;
}

int selected(bool condition, int *first, int *second) {
  ++(condition ? first : second);
  (condition ? first : second) += 1;
  return *first * 10 + *second;
}

int *&choose(int *&pointer, int &calls) {
  ++calls;
  return pointer;
}

int amount(int &calls) { return ++calls; }

int loop(int *pointer) {
  int result = 0;
  for (int i = 0; i < 4; ++i) {
    result += *pointer++;
  }
  return result;
}

using Row = int[2];

int rows(Row *&pointer) {
  pointer += 1;
  int post = (*pointer++)[1];
  int pre = (*--pointer)[0];
  pointer -= 1;
  return post * 10 + pre;
}

int main() {
  int values[6] = {1, 2, 3, 4, 5, 6};
  int other[3] = {6, 7, 8};
  int *pointer = values;
  if (steps(pointer, 1) != 2442 || *pointer != 1) {
    return 1;
  }
  pointer += 4;
  if (steps(pointer, -2) != 3553 || *pointer != 5) {
    return 2;
  }
  if (selected(true, values, other) != 36 ||
      selected(false, values, other) != 18) {
    return 3;
  }
  pointer = values;
  int calls = 0;
  choose(pointer, calls) += amount(calls);
  if (calls != 2 || *pointer != 2) {
    return 4;
  }
  int post = *choose(pointer, calls)++;
  if (calls != 3 || post != 2 || *pointer != 3) {
    return 5;
  }
  int *&alias = ++pointer;
  (alias += 1) -= 2;
  if (*pointer != 3 || loop(values) != 10) {
    return 6;
  }
  unsigned char positive = 255;
  int large[257] = {};
  large[255] = 17;
  pointer = large;
  pointer += positive;
  signed char negative = -1;
  pointer -= negative;
  *pointer = 19;
  if (large[256] != 19 || *--pointer != 17) {
    return 7;
  }
  Row matrix[3] = {{1, 2}, {3, 4}, {5, 6}};
  Row *row = matrix;
  if (rows(row) != 43 || (*row)[1] != 2) {
    return 8;
  }
  double wide[3] = {1.5, 2.5, 3.5};
  double *floating = wide;
  floating += 1;
  double old = *floating++;
  if (old != 2.5 || *floating != 3.5) {
    return 9;
  }
  return 0;
}
