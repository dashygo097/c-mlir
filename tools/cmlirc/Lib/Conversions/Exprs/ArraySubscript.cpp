#include "../../Converter.h"

namespace cmlirc {
mlir::Value
CMLIRConverter::generateArraySubscriptExpr(clang::ArraySubscriptExpr *expr) {
  mlir::OpBuilder &builder = context_manager_.Builder();
  mlir::Location loc = builder.getUnknownLoc();

  llvm::SmallVector<mlir::Value, 4> indices;
  clang::Expr *currentExpr = expr;

  while (auto *arraySubscript =
             llvm::dyn_cast<clang::ArraySubscriptExpr>(currentExpr)) {
    mlir::Value idx = generateExpr(arraySubscript->getIdx());
    if (!idx) {
      llvm::errs() << "Failed to generate index\n";
      return nullptr;
    }

    indices.insert(indices.begin(), idx);
    currentExpr = arraySubscript->getBase()->IgnoreImpCasts();
  }

  mlir::Value base = generateExpr(currentExpr);
  if (!base) {
    llvm::errs() << "Failed to generate base\n";
    return nullptr;
  }

  llvm::SmallVector<mlir::Value, 4> indexValues;
  for (mlir::Value idx : indices) {
    auto indexValue = mlir::arith::IndexCastOp::create(
                          builder, loc, builder.getIndexType(), idx)
                          .getResult();
    indexValues.push_back(indexValue);
  }

  lastArrayAccess_ = ArrayAccessInfo{base, indexValues};

  return base;
}

} // namespace cmlirc
