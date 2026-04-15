#include "../../../Converter.h"
#include "../../Utils/LHS.h"
#include "../../Utils/Numerics.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "llvm/Support/WithColor.h"

namespace cmlirc {

// Compute `value ± 1` for integer or float types.
auto applyIncDec(mlir::OpBuilder &builder, mlir::Location loc,
                 mlir::Value value, bool isIncrement) -> mlir::Value {
  if (mlir::isa<mlir::IntegerType>(value.getType())) {
    return isIncrement ? utils::addi(builder, loc, value, 1)
                       : utils::subi(builder, loc, value, 1);
  }
  if (mlir::isa<mlir::FloatType>(value.getType())) {
    return isIncrement ? utils::addf(builder, loc, value, 1.0)
                       : utils::subf(builder, loc, value, 1.0);
  }
  llvm::WithColor::error()
      << "cmlirc: unsupported type for increment/decrement\n";
  return nullptr;
}

auto CMLIRConverter::generateIncDecUnaryOperator(clang::Expr *expr,
                                                 bool isIncrement,
                                                 bool isPrefix) -> mlir::Value {
  mlir::OpBuilder &builder = contextManager.Builder();
  mlir::Location loc = builder.getUnknownLoc();

  // Fast path with SSA parameter
  clang::Expr *bare = expr->IgnoreParenImpCasts();
  if (auto *declRef = mlir::dyn_cast<clang::DeclRefExpr>(bare)) {
    if (auto *parm = mlir::dyn_cast<clang::ParmVarDecl>(declRef->getDecl())) {
      auto it = paramTable.find(parm);
      if (it != paramTable.end()) {
        mlir::Value oldVal = it->second;
        mlir::Value newVal = applyIncDec(builder, loc, oldVal, isIncrement);
        if (!newVal) {
          return nullptr;
        }
        it->second = newVal;
        return isPrefix ? newVal : oldVal;
      }
    }
  }

  // General path with lvalue (Scalar, Indexed, or Member)
  utils::LHSKind lhsKind = utils::classifyLHS(expr);

  mlir::Value lhsAddr = generateExpr(expr);
  if (!lhsAddr) {
    llvm::WithColor::error()
        << "cmlirc: cannot get lvalue for increment/decrement\n";
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

  // Load → compute → store
  mlir::Value oldVal =
      utils::loadLHS(builder, loc, lhsKind, lhsAddr, arrayAccess, elementType);

  if (!oldVal) {
    return nullptr;
  }

  mlir::Value newVal = applyIncDec(builder, loc, oldVal, isIncrement);
  if (!newVal) {
    return nullptr;
  }

  utils::storeLHS(builder, loc, lhsKind, newVal, lhsAddr, arrayAccess);

  return isPrefix ? newVal : oldVal;
}

} // namespace cmlirc
