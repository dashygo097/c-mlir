// RUN: split-file %s %t
// RUN: not clang-to-mlir %t/reference.cpp -- -std=c++17 2>&1 | FileCheck %s --check-prefix=REFERENCE
// RUN: not clang-to-mlir %t/virtual.cpp -- -std=c++17 2>&1 | FileCheck %s --check-prefix=VIRTUAL
// RUN: not clang-to-mlir %t/bitfield.cpp -- -std=c++17 2>&1 | FileCheck %s --check-prefix=BITFIELD
// RUN: not clang-to-mlir %t/indirect.cpp -- -std=c++17 2>&1 | FileCheck %s --check-prefix=INDIRECT
//--- reference.cpp
int reference(int &n) { return ++n; }

// REFERENCE: reference bindings are unsupported
//--- virtual.cpp
struct Virtual {
  virtual int value() { return 1; }
};

// VIRTUAL: virtual methods are unsupported
//--- bitfield.cpp
struct Bits {
  unsigned value : 3;
};

unsigned bitfield(Bits bits) { return bits.value; }

// BITFIELD: bitfield layouts are unsupported
//--- indirect.cpp
void indirect(void (*function)()) { function(); }

// INDIRECT: indirect calls are unsupported
