#include "cmlir/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/raw_ostream.h"

namespace cmlir {

#define GEN_PASS_DEF_CONSTANTMERGEPASS
#include "cmlir/Transforms/Passes.h.inc"

namespace {

struct ConstantKey {
  mlir::Type type;
  mlir::Attribute value;

  bool operator==(const ConstantKey &other) const {
    return type == other.type && value == other.value;
  }
};

} // namespace

} // namespace cmlir

namespace llvm {
template <> struct DenseMapInfo<cmlir::ConstantKey> {
  static inline cmlir::ConstantKey getEmptyKey() {
    return {DenseMapInfo<mlir::Type>::getEmptyKey(),
            DenseMapInfo<mlir::Attribute>::getEmptyKey()};
  }

  static inline cmlir::ConstantKey getTombstoneKey() {
    return {DenseMapInfo<mlir::Type>::getTombstoneKey(),
            DenseMapInfo<mlir::Attribute>::getTombstoneKey()};
  }

  static unsigned getHashValue(const cmlir::ConstantKey &key) {
    return llvm::hash_combine(
        DenseMapInfo<mlir::Type>::getHashValue(key.type),
        DenseMapInfo<mlir::Attribute>::getHashValue(key.value));
  }

  static bool isEqual(const cmlir::ConstantKey &lhs,
                      const cmlir::ConstantKey &rhs) {
    return lhs == rhs;
  }
};
} // namespace llvm

namespace cmlir {

namespace {

struct ConstantMergePass
    : public impl::ConstantMergePassBase<ConstantMergePass> {

  void runOnOperation() override {
    auto funcOp = getOperation();

    llvm::outs() << "\n=== Running Constant Merge Pass on function: "
                 << funcOp.getName() << " ===\n";

    llvm::DenseMap<ConstantKey, mlir::arith::ConstantOp> constantMap;

    llvm::SmallVector<mlir::arith::ConstantOp, 4> toErase;

    int mergedCount = 0;

    funcOp.walk([&](mlir::arith::ConstantOp constOp) {
      ConstantKey key{constOp.getType(), constOp.getValue()};

      auto it = constantMap.find(key);
      if (it != constantMap.end()) {
        mlir::arith::ConstantOp existingConst = it->second;

        llvm::outs() << "  Merging duplicate constant: ";
        constOp.print(llvm::outs());
        llvm::outs() << "\n    with: ";
        existingConst.print(llvm::outs());
        llvm::outs() << "\n";

        constOp.replaceAllUsesWith(existingConst.getResult());

        toErase.push_back(constOp);
        mergedCount++;

      } else {
        constantMap[key] = constOp;
      }
    });

    for (auto constOp : toErase) {
      constOp.erase();
    }

    llvm::outs() << "=== Merged " << mergedCount
                 << " duplicate constants ===\n\n";
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createConstantMergePass() {
  return std::make_unique<ConstantMergePass>();
}

} // namespace cmlir
