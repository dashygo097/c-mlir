#include "../../Converter.h"
#include "../Utils/Casts.h"
#include "llvm/Support/WithColor.h"

namespace cmlirc {

auto CMLIRConverter::generateArraySubscriptExpr(
    clang::ArraySubscriptExpr *arraySub) -> mlir::Value {
  mlir::OpBuilder &builder = contextManager.Builder();
  mlir::Location loc = builder.getUnknownLoc();

  llvm::SmallVector<mlir::Value, 4> indices;
  clang::Expr *currentExpr = arraySub;

  while (auto *arraySubscript =
             mlir::dyn_cast<clang::ArraySubscriptExpr>(currentExpr)) {
    mlir::Value idx = generateExpr(arraySubscript->getIdx());
    if (!idx) {
      llvm::WithColor::error() << "cmlirc: failed to generate index\n";
      return nullptr;
    }

    indices.insert(indices.begin(), idx);

    clang::Expr *base = arraySubscript->getBase();
    if (auto *implCast = mlir::dyn_cast<clang::ImplicitCastExpr>(base)) {
      if (implCast->getCastKind() == clang::CK_ArrayToPointerDecay) {
        base = implCast->getSubExpr();
      }
    }
    currentExpr = base;
  }

  mlir::Value base = generateExpr(currentExpr);
  if (!base) {
    llvm::WithColor::error() << "cmlirc: failed to generate base\n";
    return nullptr;
  }

  llvm::SmallVector<mlir::Value, 4> indexValues;
  for (mlir::Value idx : indices) {
    auto indexValue = detail::toIndex(builder, loc, idx);
    indexValues.push_back(indexValue);
  }

  lastArrayAccess = ArrayAccessInfo{base, indexValues};

  return base;
}

} // namespace cmlirc
