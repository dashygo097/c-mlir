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

int &marked(int &visits, int &value) {
  ++visits;
  return value;
}

void touch(int &visits) { ++visits; }

void void_return(int &visits) { return touch(visits); }

int discarded(bool choose) {
  int visits = 0;
  int first, second;
  marked(visits, first);
  choose ? marked(visits, first) : marked(visits, second);
  (void)marked(visits, second);
  (void)first;
  void_return(visits);
  return visits;
}

int discarded_index() {
  int values[2];
  int index = 0;
  (void)values[index++];
  return index;
}

int void_increment() {
  int visits = 0;
  for (int i = 0; i < 3; touch(visits)) {
    ++i;
  }
  return visits;
}

int main() {
  if (discarded(true) != 4 || discarded(false) != 4) {
    return 1;
  }
  if (discarded_index() != 1 || void_increment() != 3) {
    return 2;
  }
  int visits = 0, value = 7;
  int read = marked(visits, value);
  if (read != 7 || visits != 1) {
    return 3;
  }
  return 0;
}
