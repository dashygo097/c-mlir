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

int assign(bool condition, int x, int y) {
  (condition ? x : y) = 3;
  return x + y;
}

int increment(bool condition, int x, int y) {
  return ++(condition ? x : y);
}

int postfix(bool condition) {
  int x = 1, y = 2;
  int old = (condition ? x : y)++;
  return old * 100 + x * 10 + y;
}

int compound(bool condition) {
  int x = 4, y = 5;
  (condition ? x : y) += 3;
  return x * 10 + y;
}

int nested(bool condition) {
  int x = 1, y = 2;
  int result = ++((condition ? x : y) = 3);
  return result * 100 + x * 10 + y;
}

int binding(bool condition) {
  int x = 1, y = 2;
  int &alias = ++((condition ? x : y) = 3);
  alias = 9;
  return x * 10 + y;
}

int indexed(bool condition) {
  int values[3] = {};
  int first = 0, second = 1;
  ++(condition ? values[first++] : values[second++]);
  return first * 100 + second * 10 + values[0] + values[1];
}

int sequenced() {
  int count = 0, x = 1, y = 2;
  int result = ((++count == 2) ? x : y) = ++count * 5;
  return result * 1000 + count * 100 + x * 10 + y;
}

int main() {
  for (int x = -4; x <= 4; ++x) {
    for (int y = -4; y <= 4; ++y) {
      if (assign(true, x, y) != 3 + y || assign(false, x, y) != x + 3 ||
          increment(true, x, y) != x + 1 || increment(false, x, y) != y + 1) {
        return 1;
      }
    }
  }
  if (postfix(true) != 122 || postfix(false) != 213 ||
      compound(true) != 75 || compound(false) != 48) {
    return 2;
  }
  if (nested(true) != 442 || nested(false) != 414 ||
      binding(true) != 92 || binding(false) != 19) {
    return 3;
  }
  if (indexed(true) != 111 || indexed(false) != 21 || sequenced() != 5252) {
    return 4;
  }
  return 0;
}
