// RUN: clang-to-mlir %s -o %t.mlir -- -std=c++17
// RUN: mlir-opt --convert-scf-to-cf --convert-to-llvm --reconcile-unrealized-casts %t.mlir -o %t.llvm.mlir
// RUN: mlir-translate --mlir-to-llvmir %t.llvm.mlir -o %t.ll
// RUN: lli %t.ll
// RUN: clang-to-mlir --disable-opt %s -o %t.raw.mlir -- -std=c++17
// RUN: mlir-opt --expand-strided-metadata --convert-scf-to-cf --convert-to-llvm --reconcile-unrealized-casts %t.raw.mlir -o %t.raw.llvm.mlir
// RUN: mlir-translate --mlir-to-llvmir %t.raw.llvm.mlir -o %t.raw.ll
// RUN: lli %t.raw.ll

enum Number { Zero,
              Negative = -3,
              Next,
              Computed = (1 << 5) + 7 };
enum class Byte : unsigned char { Max = 255 };
enum class SignedByte : signed char { Negative = -3 };
enum class Signed128 : __int128 { Negative = -1,
                                  Wide = -((__int128)1 << 100) };
enum class Unsigned128 : unsigned __int128 {
  Wide = (unsigned __int128)1 << 100,
  Next
};

int promote_unsigned(Byte value) { return (int)value; }

int promote_signed(SignedByte value) { return (int)value; }

Signed128 negative(void) { return Signed128::Negative; }

Signed128 wide_signed(void) { return Signed128::Wide; }

Unsigned128 next_unsigned(void) { return Unsigned128::Next; }

int main(void) {
  if (Zero != 0 || Negative != -3 || Next != -2 || Computed != 39) {
    return 1;
  }
  if (promote_unsigned(Byte::Max) != 255 ||
      promote_signed(SignedByte::Negative) != -3) {
    return 2;
  }
  if ((__int128)negative() != -1) {
    return 3;
  }
  if ((__int128)wide_signed() != -((__int128)1 << 100)) {
    return 4;
  }
  if ((unsigned __int128)next_unsigned() !=
      ((unsigned __int128)1 << 100) + 1) {
    return 5;
  }
  return 0;
}
