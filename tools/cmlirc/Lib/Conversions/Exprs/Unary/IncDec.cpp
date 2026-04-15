#include "../../../Converter.h"
#include "../../Utils/Numerics.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "clang/AST/OperationKinds.h"
#include "llvm/Support/WithColor.h"

namespace cmlirc {
// True when `expr` is an LHS that resolves to a memref + indices pair
auto isIndexedLValue(clang::Expr *expr) -> bool {
  clang::Expr *bare = expr->IgnoreParenImpCasts();
  if (mlir::isa<clang::ArraySubscriptExpr>(bare)) {
    return true;
  }
  if (auto *uo = mlir::dyn_cast<clang::UnaryOperator>(bare)) {
    return uo->getOpcode() == clang::UO_Deref;
  }
  return false;
}

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

  // General path with lvalue backed by a memref
  bool needsArrayAccess = isIndexedLValue(expr);

  mlir::Value memref = generateExpr(expr);
  if (!memref) {
    llvm::WithColor::error()
        << "cmlirc: cannot get lvalue for increment/decrement\n";
    return nullptr;
  }

  std::optional<ArrayAccessInfo> access;
  if (needsArrayAccess) {
    if (!lastArrayAccess) {
      llvm::WithColor::error() << "cmlirc: array access info not available\n";
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
  if (!newVal) {
    return nullptr;
  }

  if (access) {
    mlir::memref::StoreOp::create(builder, loc, newVal, access->base,
                                  access->indices);
  } else {
    mlir::memref::StoreOp::create(builder, loc, newVal, memref,
                                  mlir::ValueRange{});
  }

  return isPrefix ? newVal : oldVal;
}

} // namespace cmlirc
