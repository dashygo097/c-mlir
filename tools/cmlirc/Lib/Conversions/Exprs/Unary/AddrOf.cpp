#include "../../../Converter.h"
#include "clang/AST/OperationKinds.h"

namespace cmlirc {
mlir::Value CMLIRConverter::generateAddrOfUnaryOperator(clang::Expr *subExpr) {
  mlir::OpBuilder &builder = context_manager_.Builder();
  mlir::Location loc = builder.getUnknownLoc();
  clang::Expr *bare = subExpr->IgnoreParenImpCasts();

  // &(*p)  →  p
  if (auto *uo = mlir::dyn_cast<clang::UnaryOperator>(bare)) {
    if (uo->getOpcode() == clang::UO_Deref) {
      mlir::Value inner = generateExpr(uo->getSubExpr());
      lastArrayAccess.reset();
      return inner;
    }
  }

  // &arr[i]  →  subview of the underlying memref
  if (mlir::isa<clang::ArraySubscriptExpr>(bare)) {
    mlir::Value base = generateExpr(subExpr);
    if (!base || !lastArrayAccess) {
      llvm::errs() << "cmlirc: AddrOf array: access info not available\n";
      return nullptr;
    }
    ArrayAccessInfo access = std::move(*lastArrayAccess);
    lastArrayAccess.reset();

    auto srcType = mlir::dyn_cast<mlir::MemRefType>(access.base.getType());
    if (!srcType)
      return nullptr;

    int64_t rank = srcType.getRank();
    llvm::SmallVector<mlir::OpFoldResult> offsets(rank), sizes(rank),
        strides(rank);
    for (int64_t i = 0; i < rank; ++i) {
      offsets[i] = i < (int64_t)access.indices.size()
                       ? mlir::OpFoldResult(access.indices[i])
                       : mlir::OpFoldResult(builder.getIndexAttr(0));
      sizes[i] = builder.getIndexAttr(1);
      strides[i] = builder.getIndexAttr(1);
    }
    auto resultType = mlir::MemRefType::get({}, srcType.getElementType());
    return mlir::memref::SubViewOp::create(builder, loc, resultType,
                                           access.base, offsets, sizes, strides)
        .getResult();
  }

  // &var  →  the memref slot directly
  if (auto *declRef = mlir::dyn_cast<clang::DeclRefExpr>(bare)) {
    if (auto *parm = mlir::dyn_cast<clang::ParmVarDecl>(declRef->getDecl())) {
      auto it = paramTable.find(parm);
      if (it != paramTable.end()) {
        // Parameter is an SSA value; spill it to a stack slot.
        auto slotType = mlir::MemRefType::get({}, it->second.getType());
        mlir::Value slot =
            mlir::memref::AllocaOp::create(builder, loc, slotType).getResult();
        mlir::memref::StoreOp::create(builder, loc, it->second, slot,
                                      mlir::ValueRange{});
        return slot;
      }
    }
    if (auto *var = mlir::dyn_cast<clang::VarDecl>(declRef->getDecl())) {
      auto it = symbolTable.find(var);
      if (it != symbolTable.end())
        return it->second;
    }
  }

  return nullptr;
}

} // namespace cmlirc
