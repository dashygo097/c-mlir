#include "../../../Converter.h"
#include "../../Utils/Casts.h"
#include "../../Utils/Constants.h"
#include "../../Utils/LHS.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "clang/AST/OperationKinds.h"
#include "llvm/Support/WithColor.h"

namespace cmlirc {

auto CMLIRConverter::generateAddrOfUnaryOperator(clang::Expr *addrofOp)
    -> mlir::Value {
  mlir::OpBuilder &builder = contextManager.Builder();
  mlir::Location loc = builder.getUnknownLoc();
  clang::Expr *bare = addrofOp->IgnoreParenImpCasts();

  if (auto *uo = mlir::dyn_cast<clang::UnaryOperator>(bare)) {
    if (uo->getOpcode() == clang::UO_Deref) {
      mlir::Value inner = generateExpr(uo->getSubExpr());
      lastArrayAccess.reset();
      return inner;
    }
  }

  if (mlir::isa<clang::ArraySubscriptExpr>(bare)) {
    mlir::Value base = generateExpr(bare);
    if (!base || !lastArrayAccess) {
      llvm::WithColor::error() << "cmlirc: access info for address-of array "
                                  "subscript not available\n";
      return nullptr;
    }

    ArrayAccessInfo access = std::move(*lastArrayAccess);
    lastArrayAccess.reset();

    if (mlir::isa<mlir::LLVM::LLVMPointerType>(access.base.getType())) {
      mlir::Type elementType = convertType(bare->getType());
      if (!elementType) {
        return nullptr;
      }

      return utils::getLLVMOffsetPointer(builder, loc, access.base, elementType,
                                         access.indices);
    }

    auto srcType = mlir::dyn_cast<mlir::MemRefType>(access.base.getType());
    if (!srcType) {
      llvm::WithColor::error()
          << "cmlirc: unsupported address-of array base type: "
          << access.base.getType() << "\n";
      return nullptr;
    }

    int64_t rank = srcType.getRank();
    llvm::SmallVector<mlir::OpFoldResult> offsets(rank);
    llvm::SmallVector<mlir::OpFoldResult> sizes(rank);
    llvm::SmallVector<mlir::OpFoldResult> strides(rank);

    for (int64_t i = 0; i < rank; ++i) {
      offsets[i] = i < static_cast<int64_t>(access.indices.size())
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

  if (auto *declRef = mlir::dyn_cast<clang::DeclRefExpr>(bare)) {
    if (auto *parm = mlir::dyn_cast<clang::ParmVarDecl>(declRef->getDecl())) {
      auto it = paramTable.find(parm);
      if (it != paramTable.end()) {
        if (parm->getType()->isPointerType()) {
          auto ptrType = mlir::LLVM::LLVMPointerType::get(builder.getContext());
          auto one = utils::intConst(builder, loc, builder.getI64Type(), 1);

          mlir::Value slot =
              mlir::LLVM::AllocaOp::create(builder, loc, ptrType, ptrType, one)
                  .getResult();

          mlir::Value value = it->second;
          if (!mlir::isa<mlir::LLVM::LLVMPointerType>(value.getType())) {
            value = utils::toPointer(builder, loc, value, ptrType);
          }

          if (!value) {
            return nullptr;
          }

          mlir::LLVM::StoreOp::create(builder, loc, value, slot);
          return slot;
        }

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
      if (it != symbolTable.end()) {
        return it->second;
      }
    }
  }

  llvm::WithColor::error() << "cmlirc: unsupported address-of operand: "
                           << bare->getStmtClassName() << "\n";
  return nullptr;
}

} // namespace cmlirc
