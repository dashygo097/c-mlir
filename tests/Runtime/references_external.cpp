// RUN: split-file %s %t
// RUN: clang -std=c++17 -emit-llvm -c %t/native.cpp -o %t.native.bc
// RUN: clang-to-mlir --disable-opt %t/main.cpp -o %t.raw.mlir -- -std=c++17
// RUN: mlir-opt --expand-strided-metadata --lower-affine --convert-scf-to-cf --convert-to-llvm --reconcile-unrealized-casts %t.raw.mlir -o %t.raw.llvm.mlir
// RUN: mlir-translate --mlir-to-llvmir %t.raw.llvm.mlir -o %t.raw.ll
// RUN: lli --extra-module=%t.native.bc %t.raw.ll
// RUN: clang-to-mlir %t/main.cpp -o %t.mlir -- -std=c++17
// RUN: mlir-opt --lower-affine --convert-scf-to-cf --convert-to-llvm --reconcile-unrealized-casts %t.mlir -o %t.llvm.mlir
// RUN: mlir-translate --mlir-to-llvmir %t.llvm.mlir -o %t.ll
// RUN: lli --extra-module=%t.native.bc %t.ll

//--- native.cpp
struct Pair {
  int first, second;
};

int &native_identity(int &value) { return value; }

const Pair &native_pair(const Pair &value) { return value; }

bool same_address(const int &first, const int &second) { return &first == &second; }

extern "C" int &cmlir_identity(int &);
extern "C" void cmlir_rebind(int *&, int *);

extern "C" void native_update(int &value) { cmlir_identity(value) = 17; }

extern "C" int native_rebind_test() {
  int first = 1, second = 2;
  int *pointer = &first;
  cmlir_rebind(pointer, &second);
  return *pointer;
}

//--- main.cpp
struct Pair {
  int first, second;
};

int &native_identity(int &);
const Pair &native_pair(const Pair &);
bool same_address(const int &, const int &);
extern "C" void native_update(int &);
extern "C" int native_rebind_test();

extern "C" int &cmlir_identity(int &value) { return value; }

extern "C" void cmlir_rebind(int *&pointer, int *replacement) {
  pointer = replacement;
}

int &identity(int &value) { return value; }

int main() {
  int value = 2;
  native_identity(value) = 9;
  if (value != 9 || !same_address(identity(value), value)) {
    return 1;
  }
  native_update(value);
  Pair pair = {4, 5};
  if (value != 17 || native_pair(pair).second != 5 || native_rebind_test() != 2) {
    return 2;
  }
  return 0;
}
