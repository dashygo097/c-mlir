// RUN: split-file %s %t
// RUN: not clang-to-mlir --disable-opt %t/bitfield.cpp -- -std=c++17 2>&1 | FileCheck %s --check-prefix=BITFIELD
// RUN: not clang-to-mlir --disable-opt %t/reference_member.cpp -- -std=c++17 2>&1 | FileCheck %s --check-prefix=MEMBER

//--- bitfield.cpp
struct Bits {
  unsigned value : 3;
};

unsigned bitfield(Bits bits) { return bits.value; }

// BITFIELD: bitfield layouts are unsupported

//--- reference_member.cpp
struct Object {
  int &value;
};

int member(Object object) { return object.value; }

// MEMBER: reference member layouts are unsupported
