#include "../../../Converter.h"
#include "../../Utils/Casts.h"
#include "../../Utils/Constants.h"
#include "../../Utils/LHS.h"
#include "../../Utils/Numerics.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "clang/AST/OperationKinds.h"
#include "llvm/Support/WithColor.h"

namespace cmlirc {

auto CMLIRConverter::generateUnaryOperator(clang::UnaryOperator *unOp)
    -> mlir::Value {
  mlir::OpBuilder &builder = contextManager.Builder();
  mlir::Location loc = builder.getUnknownLoc();
  clang::Expr *subExpr = unOp->getSubExpr();

  using CUO = clang::UnaryOperatorKind;

  switch (unOp->getOpcode()) {
  case CUO::UO_Plus:
    return generateExpr(subExpr);

  case CUO::UO_Minus: {
    mlir::Value value = generateExpr(subExpr);
    return utils::neg(builder, loc, value);
  }

  case CUO::UO_PreInc:
    return generateIncDecUnaryOperator(subExpr, true, true);

  case CUO::UO_PostInc:
    return generateIncDecUnaryOperator(subExpr, true, false);

  case CUO::UO_PreDec:
    return generateIncDecUnaryOperator(subExpr, false, true);

  case CUO::UO_PostDec:
    return generateIncDecUnaryOperator(subExpr, false, false);

  case CUO::UO_LNot: {
    mlir::Value value = generateExpr(subExpr);
    value = utils::toBool(builder, loc, value);

    if (!value) {
      return nullptr;
    }

    mlir::Value one = utils::boolConst(builder, loc, true);
    return mlir::arith::XOrIOp::create(builder, loc, value, one).getResult();
  }

  case CUO::UO_Not: {
    mlir::Value value = generateExpr(subExpr);
    return utils::bitNot(builder, loc, value);
  }

  case CUO::UO_Deref: {
    mlir::Value base = generateExpr(subExpr);
    if (!base) {
      return nullptr;
    }

    if (lastArrayAccess && lastArrayAccess->base == base) {
      ArrayAccessInfo access = std::move(*lastArrayAccess);
      lastArrayAccess.reset();

      if (mlir::isa<mlir::LLVM::LLVMPointerType>(access.base.getType())) {
        auto ptrType = mlir::LLVM::LLVMPointerType::get(builder.getContext());

        mlir::Value slotPtr = utils::getLLVMOffsetPointer(
            builder, loc, access.base, ptrType, access.indices);

        base = mlir::LLVM::LoadOp::create(builder, loc, ptrType, slotPtr)
                   .getResult();
      } else if (mlir::isa<mlir::MemRefType>(access.base.getType())) {
        base = mlir::memref::LoadOp::create(builder, loc, access.base,
                                            access.indices)
                   .getResult();
      } else {
        llvm::WithColor::error()
            << "cmlirc: unsupported nested dereference base type: "
            << access.base.getType() << "\n";
        return nullptr;
      }

      if (!base) {
        return nullptr;
      }
    } else {
      clang::Expr *bareSubExpr = subExpr->IgnoreParenImpCasts();

      if (auto *declRef = mlir::dyn_cast<clang::DeclRefExpr>(bareSubExpr)) {
        if (auto *varDecl =
                mlir::dyn_cast<clang::VarDecl>(declRef->getDecl())) {
          if (!mlir::isa<clang::ParmVarDecl>(varDecl) &&
              varDecl->getType()->isPointerType() &&
              symbolTable.count(varDecl) && symbolTable[varDecl] == base &&
              mlir::isa<mlir::LLVM::LLVMPointerType>(base.getType())) {
            auto ptrType =
                mlir::LLVM::LLVMPointerType::get(builder.getContext());

            base = mlir::LLVM::LoadOp::create(builder, loc, ptrType, base)
                       .getResult();
          }
        }
      }
    }

    if (auto memrefTy = mlir::dyn_cast<mlir::MemRefType>(base.getType())) {
      if (memrefTy.getRank() == 0) {
        return base;
      }

      lastArrayAccess =
          ArrayAccessInfo{base, {utils::indexConst(builder, loc, 0)}};
      return base;
    }

    if (mlir::isa<mlir::LLVM::LLVMPointerType>(base.getType())) {
      lastArrayAccess = ArrayAccessInfo{
          base, {utils::intConst(builder, loc, builder.getI64Type(), 0)}};
      return base;
    }

    llvm::WithColor::error() << "cmlirc: cannot dereference value of type "
                             << base.getType() << "\n";
    return nullptr;
  }

  case CUO::UO_AddrOf:
    return generateAddrOfUnaryOperator(subExpr);

  default:
    llvm::WithColor::error()
        << "cmlirc: unsupported unary operator: "
        << clang::UnaryOperator::getOpcodeStr(unOp->getOpcode()) << "\n";
    return nullptr;
  }
}

} // namespace cmlirc
