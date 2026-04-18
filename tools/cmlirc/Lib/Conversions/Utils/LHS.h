#ifndef CMLIRC_LHS_H
#define CMLIRC_LHS_H

#include "../../Converter.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "llvm/Support/WithColor.h"

namespace cmlirc::utils {
enum class LHSKind { Scalar, Indexed, Member };

inline auto classifyLHS(clang::Expr *lhs) -> LHSKind {
  clang::Expr *bare = lhs->IgnoreParenImpCasts();
  if (mlir::isa<clang::ArraySubscriptExpr>(bare)) {
    return LHSKind::Indexed;
  }
  if (auto *uo = mlir::dyn_cast<clang::UnaryOperator>(bare)) {
    if (uo->getOpcode() == clang::UO_Deref) {
      return LHSKind::Indexed;
    }
  }
  if (mlir::isa<clang::MemberExpr>(bare)) {
    return LHSKind::Member;
  }
  return LHSKind::Scalar;
}

inline auto getLLVMOffsetPointer(mlir::OpBuilder &builder, mlir::Location loc,
                                 mlir::Value basePtr, mlir::Type elementType,
                                 llvm::ArrayRef<mlir::Value> indices)
    -> mlir::Value {
  llvm::SmallVector<mlir::LLVM::GEPArg> gepIndices;

  for (mlir::Value idx : indices) {
    if (idx.getType().isIndex()) {
      idx = mlir::arith::IndexCastOp::create(builder, loc, builder.getI64Type(),
                                             idx)
                .getResult();
    }
    gepIndices.push_back(idx);
  }

  auto ptrType = mlir::LLVM::LLVMPointerType::get(builder.getContext());
  return mlir::LLVM::GEPOp::create(builder, loc, ptrType, elementType, basePtr,
                                   gepIndices)
      .getResult();
}

inline auto loadLHS(mlir::OpBuilder &builder, mlir::Location loc, LHSKind kind,
                    mlir::Value lhsMemref,
                    const std::optional<cmlirc::ArrayAccessInfo> &arrayAccess,
                    mlir::Type elementType) -> mlir::Value {
  switch (kind) {
  case LHSKind::Indexed:
    if (mlir::isa<mlir::LLVM::LLVMPointerType>(arrayAccess->base.getType())) {
      mlir::Value offsetPtr = getLLVMOffsetPointer(
          builder, loc, arrayAccess->base, elementType, arrayAccess->indices);
      return mlir::LLVM::LoadOp::create(builder, loc, elementType, offsetPtr)
          .getResult();
    }
    return mlir::memref::LoadOp::create(builder, loc, arrayAccess->base,
                                        arrayAccess->indices)
        .getResult();
  case LHSKind::Member:
    return mlir::LLVM::LoadOp::create(builder, loc, elementType, lhsMemref);
  case LHSKind::Scalar:
    return mlir::memref::LoadOp::create(builder, loc, lhsMemref).getResult();
  }

  llvm::WithColor::error() << "cmlirc: unhandled LHSKind\n";
  return nullptr;
}

inline void
storeLHS(mlir::OpBuilder &builder, mlir::Location loc, LHSKind kind,
         mlir::Value value, mlir::Value lhsMemref,
         const std::optional<cmlirc::ArrayAccessInfo> &arrayAccess) {
  switch (kind) {
  case LHSKind::Indexed:
    if (mlir::isa<mlir::LLVM::LLVMPointerType>(arrayAccess->base.getType())) {
      mlir::Value offsetPtr =
          getLLVMOffsetPointer(builder, loc, arrayAccess->base, value.getType(),
                               arrayAccess->indices);
      mlir::LLVM::StoreOp::create(builder, loc, value, offsetPtr);
      break;
    }
    mlir::memref::StoreOp::create(builder, loc, value, arrayAccess->base,
                                  arrayAccess->indices);
    break;
  case LHSKind::Member:
    mlir::LLVM::StoreOp::create(builder, loc, value, lhsMemref);
    break;
  case LHSKind::Scalar:
    mlir::memref::StoreOp::create(builder, loc, value, lhsMemref);
    break;
  }
}

} // namespace cmlirc::utils

#endif // CMLIRC_LHS_H
