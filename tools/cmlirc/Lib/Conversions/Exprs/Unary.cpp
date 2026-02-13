#include "../../Converter.h"

namespace cmlirc {

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
      mlir::Value zero =
          mlir::arith::ConstantOp::create(builder, loc, type,
                                          builder.getIntegerAttr(type, 0))
              .getResult();
      return mlir::arith::SubIOp::create(builder, loc, zero, operand)
          .getResult();
    } else if (mlir::isa<mlir::FloatType>(type)) {
      return mlir::arith::NegFOp::create(builder, loc, operand).getResult();
    }

    return nullptr;
  }
  case clang::UO_PreInc: {
    return generateIncrementDecrement(subExpr, /*isIncrement=*/true,
                                      /*isPrefix=*/true);
  }
  case clang::UO_PostInc: {
    return generateIncrementDecrement(subExpr, /*isIncrement=*/true,
                                      /*isPrefix=*/false);
  }
  case clang::UO_PreDec: {
    return generateIncrementDecrement(subExpr, /*isIncrement=*/false,
                                      /*isPrefix=*/true);
  }
  case clang::UO_PostDec: {
    return generateIncrementDecrement(subExpr, /*isIncrement=*/false,
                                      /*isPrefix=*/false);
  }
  case clang::UO_LNot: {
    mlir::Value operand = generateExpr(subExpr);
    if (!operand)
      return nullptr;

    mlir::Type type = operand.getType();

    if (mlir::isa<mlir::IntegerType>(type)) {
      mlir::Value zero =
          mlir::arith::ConstantOp::create(builder, loc, type,
                                          builder.getIntegerAttr(type, 0))
              .getResult();
      return mlir::arith::CmpIOp::create(
                 builder, loc, mlir::arith::CmpIPredicate::eq, operand, zero)
          .getResult();
    } else if (mlir::isa<mlir::FloatType>(type)) {
      mlir::Value zero =
          mlir::arith::ConstantOp::create(builder, loc, type,
                                          builder.getFloatAttr(type, 0.0))
              .getResult();
      return mlir::arith::CmpFOp::create(
                 builder, loc, mlir::arith::CmpFPredicate::OEQ, operand, zero)
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
      mlir::Value allOnes =
          mlir::arith::ConstantOp::create(builder, loc, type,
                                          builder.getIntegerAttr(type, -1))
              .getResult();
      return mlir::arith::XOrIOp::create(builder, loc, operand, allOnes)
          .getResult();
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

  clang::Expr *lvalueExpr = expr->IgnoreParenImpCasts();

  if (auto *declRef = llvm::dyn_cast<clang::DeclRefExpr>(lvalueExpr)) {
    if (auto *paramDecl =
            llvm::dyn_cast<clang::ParmVarDecl>(declRef->getDecl())) {
      if (paramTable.count(paramDecl)) {
        mlir::Value oldValue = paramTable[paramDecl];
        mlir::Type type = oldValue.getType();
        mlir::Value one;
        mlir::Value newValue;

        if (mlir::isa<mlir::IntegerType>(type)) {
          one = mlir::arith::ConstantOp::create(builder, loc, type,
                                                builder.getIntegerAttr(type, 1))
                    .getResult();
          newValue =
              isIncrement
                  ? mlir::arith::AddIOp::create(builder, loc, oldValue, one)
                        .getResult()
                  : mlir::arith::SubIOp::create(builder, loc, oldValue, one)
                        .getResult();
        } else if (mlir::isa<mlir::FloatType>(type)) {
          one = mlir::arith::ConstantOp::create(builder, loc, type,
                                                builder.getFloatAttr(type, 1.0))
                    .getResult();
          newValue =
              isIncrement
                  ? mlir::arith::AddFOp::create(builder, loc, oldValue, one)
                        .getResult()
                  : mlir::arith::SubFOp::create(builder, loc, oldValue, one)
                        .getResult();
        } else {
          llvm::errs() << "Unsupported type for increment/decrement\n";
          return nullptr;
        }

        paramTable[paramDecl] = newValue;

        return isPrefix ? newValue : oldValue;
      }
    }
  }

  bool isArrayAccess = llvm::isa<clang::ArraySubscriptExpr>(lvalueExpr);
  std::optional<ArrayAccessInfo> savedArrayAccess;

  mlir::Value lvalue = generateExpr(lvalueExpr);

  if (!lvalue) {
    llvm::errs() << "Cannot get lvalue for increment/decrement\n";
    return nullptr;
  }

  if (isArrayAccess) {
    if (lastArrayAccess_) {
      savedArrayAccess = lastArrayAccess_;
      lastArrayAccess_.reset();
    } else {
      llvm::errs() << "Array access info not available\n";
      return nullptr;
    }
  }

  mlir::Value oldValue;

  if (isArrayAccess && savedArrayAccess) {
    oldValue =
        mlir::memref::LoadOp::create(builder, loc, savedArrayAccess->base,
                                     savedArrayAccess->indices)
            .getResult();
  } else {
    oldValue = mlir::memref::LoadOp::create(builder, loc, lvalue).getResult();
  }

  mlir::Type type = oldValue.getType();
  mlir::Value one;
  mlir::Value newValue;

  if (mlir::isa<mlir::IntegerType>(type)) {
    one = mlir::arith::ConstantOp::create(builder, loc, type,
                                          builder.getIntegerAttr(type, 1))
              .getResult();
    newValue = isIncrement
                   ? mlir::arith::AddIOp::create(builder, loc, oldValue, one)
                         .getResult()
                   : mlir::arith::SubIOp::create(builder, loc, oldValue, one)
                         .getResult();
  } else if (mlir::isa<mlir::FloatType>(type)) {
    one = mlir::arith::ConstantOp::create(builder, loc, type,
                                          builder.getFloatAttr(type, 1.0))
              .getResult();
    newValue = isIncrement
                   ? mlir::arith::AddFOp::create(builder, loc, oldValue, one)
                         .getResult()
                   : mlir::arith::SubFOp::create(builder, loc, oldValue, one)
                         .getResult();
  } else {
    llvm::errs() << "Unsupported type for increment/decrement\n";
    return nullptr;
  }

  if (isArrayAccess && savedArrayAccess) {
    mlir::memref::StoreOp::create(builder, loc, newValue,
                                  savedArrayAccess->base,
                                  savedArrayAccess->indices);
  } else {
    mlir::memref::StoreOp::create(builder, loc, newValue, lvalue,
                                  mlir::ValueRange{});
  }

  return isPrefix ? newValue : oldValue;
}

} // namespace cmlirc
