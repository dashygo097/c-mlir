// RUN: clang-to-mlir --disable-opt %s -o %t.raw.mlir -- -std=c++17
// RUN: mlir-opt --expand-strided-metadata --convert-scf-to-cf --convert-to-llvm --reconcile-unrealized-casts %t.raw.mlir -o %t.raw.llvm.mlir
// RUN: mlir-translate --mlir-to-llvmir %t.raw.llvm.mlir -o %t.raw.ll
// RUN: lli %t.raw.ll
// RUN: clang-to-mlir %s -o %t.mlir -- -std=c++17
// RUN: mlir-opt --convert-scf-to-cf --convert-to-llvm --reconcile-unrealized-casts %t.mlir -o %t.llvm.mlir
// RUN: mlir-translate --mlir-to-llvmir %t.llvm.mlir -o %t.ll
// RUN: lli %t.ll
// RUN: clang-to-mlir --struct-to-memref %s -o %t.promoted.mlir -- -std=c++17
// RUN: mlir-opt --convert-scf-to-cf --convert-to-llvm --reconcile-unrealized-casts %t.promoted.mlir -o %t.promoted.llvm.mlir
// RUN: mlir-translate --mlir-to-llvmir %t.promoted.llvm.mlir -o %t.promoted.ll
// RUN: lli %t.promoted.ll

class Object {
public:
  int value;

  int increment() { return ++value; }
};

struct Pair {
  int first, second;
};

struct Nested {
  Pair pair;
  int values[3];
};

int next(int *count) { return ++*count; }

int record_array(Pair (*objects)[2]) { return (*objects)[1].first; }

int modify_copy(Pair value) {
  value.first = 99;
  return value.first;
}

const char *message(void) { return "ok"; }

int *identity(int *value) { return value; }

int sum_two(const int *values) { return values[0] + values[1]; }

int main(void) {
  int values[3] = {1, 2};
  int empty[2] = {};
  int matrix[2][3] = {{1}, {2, 3}};
  if (values[2] != 0 || empty[0] != 0 || empty[1] != 0 ||
      matrix[0][1] != 0 || matrix[1][2] != 0) {
    return 1;
  }
  Object object;
  object.value = 3;
  Pair pair = {4};
  Pair copy = pair;
  Nested nested = {{5}, {6}};
  Pair objects[2] = {{7}, {8}};
  if (object.value != 3 || copy.first != 4 || pair.second != 0 ||
      nested.pair.second != 0 || nested.values[2] != 0 ||
      record_array(&objects) != 8) {
    return 2;
  }
  int count = 0;
  int sequence[4] = {next(&count), next(&count)};
  if (count != 2 || sequence[0] != 1 || sequence[1] != 2 || sequence[3] != 0) {
    return 3;
  }
  unsigned __int128 wide = (unsigned __int128)1 << 100;
  if ((~wide >> 100) != (((unsigned __int128)1 << 28) - 2)) {
    return 4;
  }
  if (message()[0] != 'o' || message()[1] != 'k') {
    return 5;
  }
  if (*identity(&sequence[1]) != 2 || sum_two(&sequence[0]) != 3 ||
      sum_two(&sequence[1]) != 2) {
    return 6;
  }
  if (modify_copy(pair) != 99 || pair.first != 4 || object.increment() != 4 ||
      Pair{9, 10}.second != 10) {
    return 7;
  }
  return 0;
}
