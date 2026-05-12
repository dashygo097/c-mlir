#include "../../../Converter.h"
#include "../../Utils/LHS.h"
#include "../../Utils/Numerics.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "llvm/Support/WithColor.h"

namespace cmlirc {

auto CMLIRConverter::generateIncDecUnaryOperator(clang::Expr *expr,
                                                 bool isIncrement,
                                                 bool isPrefix) -> mlir::Value {
  mlir::OpBuilder &builder = contextManager.Builder();
  mlir::Location loc = builder.getUnknownLoc();

  clang::Expr *bare = expr->IgnoreParenImpCasts();

  if (auto *declRef = mlir::dyn_cast<clang::DeclRefExpr>(bare)) {
    if (auto *parm = mlir::dyn_cast<clang::ParmVarDecl>(declRef->getDecl())) {
      auto it = paramTable.find(parm);
      if (it != paramTable.end()) {
        mlir::Value oldVal = it->second;
        mlir::Value newVal = isIncrement ? utils::inc(builder, loc, oldVal)
                                         : utils::dec(builder, loc, oldVal);

        if (!newVal) {
          return nullptr;
        }

        it->second = newVal;
        return isPrefix ? newVal : oldVal;
      }
    }
  }

  utils::LHSKind lhsKind = utils::classifyLHS(expr);

  mlir::Value lhsAddr = generateExpr(expr);
  if (!lhsAddr) {
    return nullptr;
  }

  std::optional<ArrayAccessInfo> arrayAccess;
  if (lhsKind == utils::LHSKind::Indexed) {
    if (!lastArrayAccess) {
      llvm::WithColor::error() << "cmlirc: array access info not available\n";
      return nullptr;
    }

    arrayAccess = std::move(lastArrayAccess);
    lastArrayAccess.reset();
  }

  mlir::Type elementType = convertType(expr->getType());
  if (!elementType) {
    return nullptr;
  }

  mlir::Value oldVal =
      utils::loadLHS(builder, loc, lhsKind, lhsAddr, arrayAccess, elementType);
  if (!oldVal) {
    return nullptr;
  }

  mlir::Value newVal = isIncrement ? utils::inc(builder, loc, oldVal)
                                   : utils::dec(builder, loc, oldVal);
  if (!newVal) {
    return nullptr;
  }

  utils::storeLHS(builder, loc, lhsKind, newVal, lhsAddr, arrayAccess);
  return isPrefix ? newVal : oldVal;
}

} // namespace cmlirc
