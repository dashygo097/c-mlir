#include "../../Converter.h"
#include "../Utils/Numeric.h"
#include "clang/AST/OperationKinds.h"

namespace cmlirc {

static bool isMemrefLValueWithIndices(clang::Expr *expr) {
  clang::Expr *base = expr->IgnoreParenImpCasts();
  if (mlir::isa<clang::ArraySubscriptExpr>(base))
    return true;
  if (auto *uo = mlir::dyn_cast<clang::UnaryOperator>(base))
    return uo->getOpcode() == clang::UO_Deref;
  return false;
}

mlir::Value CMLIRConverter::generateUnaryOperator(clang::UnaryOperator *unOp) {
  mlir::OpBuilder &builder = context_manager_.Builder();
  mlir::Location loc = builder.getUnknownLoc();
  clang::Expr *subExpr = unOp->getSubExpr();

  switch (unOp->getOpcode()) {
  case clang::UO_Plus:
    return generateExpr(subExpr);

  case clang::UO_Minus: {
    mlir::Value operand = generateExpr(subExpr);
    if (!operand)
      return nullptr;
    mlir::Type type = operand.getType();
    if (mlir::isa<mlir::IntegerType>(type)) {
      return mlir::arith::SubIOp::create(
                 builder, loc, detail::intConst(builder, loc, type, 0), operand)
          .getResult();
    } else if (mlir::isa<mlir::FloatType>(type)) {
      return mlir::arith::NegFOp::create(builder, loc, operand).getResult();
    }
    return nullptr;
  }

  case clang::UO_PreInc:
    return generateIncrementDecrement(subExpr, /*isIncrement=*/true,
                                      /*isPrefix=*/true);
  case clang::UO_PostInc:
    return generateIncrementDecrement(subExpr, /*isIncrement=*/true,
                                      /*isPrefix=*/false);
  case clang::UO_PreDec:
    return generateIncrementDecrement(subExpr, /*isIncrement=*/false,
                                      /*isPrefix=*/true);
  case clang::UO_PostDec:
    return generateIncrementDecrement(subExpr, /*isIncrement=*/false,
                                      /*isPrefix=*/false);

  case clang::UO_LNot: {
    mlir::Value operand = generateExpr(subExpr);
    if (!operand)
      return nullptr;
    mlir::Type type = operand.getType();
    if (mlir::isa<mlir::IntegerType>(type)) {
      return mlir::arith::CmpIOp::create(
                 builder, loc, mlir::arith::CmpIPredicate::eq, operand,
                 detail::intConst(builder, loc, type, 0))
          .getResult();
    } else if (mlir::isa<mlir::FloatType>(type)) {
      return mlir::arith::CmpFOp::create(
                 builder, loc, mlir::arith::CmpFPredicate::OEQ, operand,
                 detail::floatConst(builder, loc, type, 0.0))
          .getResult();
    }
    return nullptr;
  }

  case clang::UO_Not: {
    mlir::Value operand = generateExpr(subExpr);
    if (!operand)
      return nullptr;
    mlir::Type type = operand.getType();
    if (mlir::isa<mlir::IntegerType>(type)) {
      return mlir::arith::XOrIOp::create(
                 builder, loc, operand,
                 detail::intConst(builder, loc, type, -1))
          .getResult();
    }
    return nullptr;
  }

  case clang::UO_Deref: {
    mlir::Value base = generateExpr(subExpr);
    if (!base)
      return nullptr;

    auto memrefType = mlir::dyn_cast<mlir::MemRefType>(base.getType());
    if (!memrefType) {
      return nullptr;
    }

    if (memrefType.getRank() == 0)
      return base;

    lastArrayAccess_ = ArrayAccessInfo{
        base, mlir::ValueRange{detail::indexConst(builder, loc, 0)}};
    return base;
  }

  case clang::UO_AddrOf: {
    clang::Expr *bare = subExpr->IgnoreParenImpCasts();

    if (auto *uo = mlir::dyn_cast<clang::UnaryOperator>(bare)) {
      if (uo->getOpcode() == clang::UO_Deref) {
        mlir::Value inner = generateExpr(uo->getSubExpr());
        lastArrayAccess_.reset();
        return inner;
      }
    }

    if (mlir::isa<clang::ArraySubscriptExpr>(bare)) {
      mlir::Value base = generateExpr(subExpr);
      if (!base)
        return nullptr;

      if (!lastArrayAccess_) {
        llvm::errs() << "AddrOf: array access info not available\n";
        return nullptr;
      }
      ArrayAccessInfo access = *lastArrayAccess_;
      lastArrayAccess_.reset();

      auto srcType = mlir::dyn_cast<mlir::MemRefType>(access.base.getType());
      if (!srcType)
        return nullptr;

      int64_t rank = srcType.getRank();
      llvm::SmallVector<mlir::OpFoldResult> offsets, sizes, strides;
      for (int64_t i = 0; i < rank; ++i) {
        offsets.push_back(i < (int64_t)access.indices.size()
                              ? mlir::OpFoldResult(access.indices[i])
                              : mlir::OpFoldResult(builder.getIndexAttr(0)));
        sizes.push_back(builder.getIndexAttr(1));
        strides.push_back(builder.getIndexAttr(1));
      }

      llvm::SmallVector<int64_t> droppedDims(rank);
      std::iota(droppedDims.begin(), droppedDims.end(), 0);

      auto resultType = mlir::MemRefType::get({}, srcType.getElementType());
      return mlir::memref::SubViewOp::create(
                 builder, loc, resultType, access.base, offsets, sizes, strides)
          .getResult();
    }

    if (auto *declRef = mlir::dyn_cast<clang::DeclRefExpr>(bare)) {
      if (auto *paramDecl =
              mlir::dyn_cast<clang::ParmVarDecl>(declRef->getDecl())) {
        if (paramTable.count(paramDecl)) {
          mlir::Value val = paramTable[paramDecl];
          auto allocType = mlir::MemRefType::get({}, val.getType());
          mlir::Value slot =
              mlir::memref::AllocaOp::create(builder, loc, allocType)
                  .getResult();
          mlir::memref::StoreOp::create(builder, loc, val, slot,
                                        mlir::ValueRange{});
          return slot;
        }
      }

      if (auto *varDecl = mlir::dyn_cast<clang::VarDecl>(declRef->getDecl())) {
        if (symbolTable.count(varDecl))
          return symbolTable[varDecl];
      }
    }

    return nullptr;
  }

  default:
    llvm::errs() << "Unsupported unary operator: "
                 << clang::UnaryOperator::getOpcodeStr(unOp->getOpcode())
                 << "\n";
    return nullptr;
  }
}

mlir::Value CMLIRConverter::generateIncrementDecrement(clang::Expr *expr,
                                                       bool isIncrement,
                                                       bool isPrefix) {
  mlir::OpBuilder &builder = context_manager_.Builder();
  mlir::Location loc = builder.getUnknownLoc();

  clang::Expr *bare = expr->IgnoreParenImpCasts();
  if (auto *declRef = mlir::dyn_cast<clang::DeclRefExpr>(bare)) {
    if (auto *paramDecl =
            mlir::dyn_cast<clang::ParmVarDecl>(declRef->getDecl())) {
      if (paramTable.count(paramDecl)) {
        mlir::Value oldValue = paramTable[paramDecl];
        mlir::Type type = oldValue.getType();
        mlir::Value newValue;

        if (mlir::isa<mlir::IntegerType>(type)) {
          newValue = isIncrement ? detail::addInt(builder, loc, oldValue, 1)
                                 : detail::subInt(builder, loc, oldValue, 1);

        } else if (mlir::isa<mlir::FloatType>(type)) {
          newValue = isIncrement
                         ? detail::addFloat(builder, loc, oldValue, 1.0)
                         : detail::subFloat(builder, loc, oldValue, 1.0);
        } else {
          llvm::errs() << "Unsupported type for increment/decrement\n";
          return nullptr;
        }

        paramTable[paramDecl] = newValue;
        return isPrefix ? newValue : oldValue;
      }
    }
  }

  bool isIndexedAccess = isMemrefLValueWithIndices(expr);

  mlir::Value memrefVal = generateExpr(expr);
  if (!memrefVal) {
    llvm::errs() << "Cannot get lvalue for increment/decrement\n";
    return nullptr;
  }

  std::optional<ArrayAccessInfo> access;
  if (isIndexedAccess) {
    if (lastArrayAccess_) {
      access = lastArrayAccess_;
      lastArrayAccess_.reset();
    } else {
      llvm::errs() << "Array access info not available\n";
      return nullptr;
    }
  }

  mlir::Value oldValue;
  if (access) {
    oldValue = mlir::memref::LoadOp::create(builder, loc, access->base,
                                            access->indices)
                   .getResult();
  } else {
    oldValue =
        mlir::memref::LoadOp::create(builder, loc, memrefVal).getResult();
  }

  mlir::Type type = oldValue.getType();
  mlir::Value newValue;

  if (mlir::isa<mlir::IntegerType>(type)) {
    newValue = isIncrement ? detail::addInt(builder, loc, oldValue, 1)
                           : detail::subInt(builder, loc, oldValue, 1);
  } else if (mlir::isa<mlir::FloatType>(type)) {
    newValue = isIncrement ? detail::addFloat(builder, loc, oldValue, 1.0)
                           : detail::subFloat(builder, loc, oldValue, 1.0);
  } else {
    llvm::errs() << "Unsupported type for increment/decrement\n";
    return nullptr;
  }

  if (access) {
    mlir::memref::StoreOp::create(builder, loc, newValue, access->base,
                                  access->indices);
  } else {
    mlir::memref::StoreOp::create(builder, loc, newValue, memrefVal,
                                  mlir::ValueRange{});
  }

  return isPrefix ? newValue : oldValue;
}

} // namespace cmlirc
