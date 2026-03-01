#include "../../../Converter.h"
#include "../../Utils/Numerics.h"
#include "clang/AST/OperationKinds.h"

namespace cmlirc {
// True when `expr` is an LHS that resolves to a memref + indices pair
bool isIndexedLValue(clang::Expr *expr) {
  clang::Expr *bare = expr->IgnoreParenImpCasts();
  if (mlir::isa<clang::ArraySubscriptExpr>(bare))
    return true;
  if (auto *uo = mlir::dyn_cast<clang::UnaryOperator>(bare))
    return uo->getOpcode() == clang::UO_Deref;
  return false;
}

// Compute `value ± 1` for integer or float types.
mlir::Value applyIncDec(mlir::OpBuilder &builder, mlir::Location loc,
                        mlir::Value value, bool isIncrement) {
  if (mlir::isa<mlir::IntegerType>(value.getType()))
    return isIncrement ? detail::addi(builder, loc, value, 1)
                       : detail::subi(builder, loc, value, 1);
  if (mlir::isa<mlir::FloatType>(value.getType()))
    return isIncrement ? detail::addf(builder, loc, value, 1.0)
                       : detail::subf(builder, loc, value, 1.0);
  llvm::errs() << "cmlirc: unsupported type for increment/decrement\n";
  return nullptr;
}

mlir::Value CMLIRConverter::generateIncDecUnaryOperator(clang::Expr *expr,
                                                        bool isIncrement,
                                                        bool isPrefix) {
  mlir::OpBuilder &builder = context_manager_.Builder();
  mlir::Location loc = builder.getUnknownLoc();

  // Fast path with SSA parameter
  clang::Expr *bare = expr->IgnoreParenImpCasts();
  if (auto *declRef = mlir::dyn_cast<clang::DeclRefExpr>(bare)) {
    if (auto *parm = mlir::dyn_cast<clang::ParmVarDecl>(declRef->getDecl())) {
      auto it = paramTable.find(parm);
      if (it != paramTable.end()) {
        mlir::Value oldVal = it->second;
        mlir::Value newVal = applyIncDec(builder, loc, oldVal, isIncrement);
        if (!newVal)
          return nullptr;
        it->second = newVal;
        return isPrefix ? newVal : oldVal;
      }
    }
  }

  // General path with lvalue backed by a memref
  bool needsArrayAccess = isIndexedLValue(expr);

  mlir::Value memref = generateExpr(expr);
  if (!memref) {
    llvm::errs() << "cmlirc: cannot get lvalue for increment/decrement\n";
    return nullptr;
  }

  std::optional<ArrayAccessInfo> access;
  if (needsArrayAccess) {
    if (!lastArrayAccess) {
      llvm::errs() << "cmlirc: array access info not available\n";
      return nullptr;
    }
    access = std::move(lastArrayAccess);
    lastArrayAccess.reset();
  }

  // Load → compute → store
  mlir::Value oldVal =
      access ? mlir::memref::LoadOp::create(builder, loc, access->base,
                                            access->indices)
                   .getResult()
             : mlir::memref::LoadOp::create(builder, loc, memref).getResult();

  mlir::Value newVal = applyIncDec(builder, loc, oldVal, isIncrement);
  if (!newVal)
    return nullptr;

  if (access)
    mlir::memref::StoreOp::create(builder, loc, newVal, access->base,
                                  access->indices);
  else
    mlir::memref::StoreOp::create(builder, loc, newVal, memref,
                                  mlir::ValueRange{});

  return isPrefix ? newVal : oldVal;
}

} // namespace cmlirc
