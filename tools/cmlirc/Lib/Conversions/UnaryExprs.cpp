#include "../../ArgumentList.h"
#include "../ASTVisitor.h"
#include "./Types.h"

namespace cmlirc {
mlir::Value CMLIRCASTVisitor::generateIncrementDecrement(clang::Expr *expr,
                                                         bool isIncrement,
                                                         bool isPrefix) {
  mlir::OpBuilder &builder = context_manager_.Builder();

  mlir::Value lvalue = generateExpr(expr, /*needLValue=*/true);
  if (!lvalue) {
    llvm::errs() << "Cannot get lvalue for increment/decrement\n";
    return nullptr;
  }

  mlir::Value oldValue;

  if (auto *arraySubscript = llvm::dyn_cast<clang::ArraySubscriptExpr>(expr)) {
    oldValue = generateArraySubscriptExpr(arraySubscript, /*needLValue=*/false);
  } else {
    oldValue =
        mlir::memref::LoadOp::create(builder, builder.getUnknownLoc(), lvalue)
            .getResult();
  }

  mlir::Type type = oldValue.getType();

  mlir::Value one;
  if (mlir::isa<mlir::IntegerType>(type)) {
    one =
        mlir::arith::ConstantOp::create(builder, builder.getUnknownLoc(), type,
                                        builder.getIntegerAttr(type, 1));
  } else if (mlir::isa<mlir::FloatType>(type)) {
    one =
        mlir::arith::ConstantOp::create(builder, builder.getUnknownLoc(), type,
                                        builder.getFloatAttr(type, 1.0));
  } else {
    llvm::errs() << "Unsupported type for increment/decrement\n";
    return nullptr;
  }

  mlir::Value newValue;
  if (mlir::isa<mlir::IntegerType>(type)) {
    if (isIncrement) {
      newValue = mlir::arith::AddIOp::create(builder, builder.getUnknownLoc(),
                                             oldValue, one)
                     .getResult();
    } else {
      newValue = mlir::arith::SubIOp::create(builder, builder.getUnknownLoc(),
                                             oldValue, one)
                     .getResult();
    }
  } else {
    if (isIncrement) {
      newValue = mlir::arith::AddFOp::create(builder, builder.getUnknownLoc(),
                                             oldValue, one)
                     .getResult();
    } else {
      newValue = mlir::arith::SubFOp::create(builder, builder.getUnknownLoc(),
                                             oldValue, one)
                     .getResult();
    }
  }

  if (auto *arraySubscript = llvm::dyn_cast<clang::ArraySubscriptExpr>(expr)) {
    mlir::Value idx = generateExpr(arraySubscript->getIdx());
    auto indexValue = mlir::arith::IndexCastOp::create(
        builder, builder.getUnknownLoc(), builder.getIndexType(), idx);

    mlir::memref::StoreOp::create(builder, builder.getUnknownLoc(), newValue,
                                  lvalue,
                                  mlir::ValueRange{indexValue.getResult()});
  } else {
    mlir::memref::StoreOp::create(builder, builder.getUnknownLoc(), newValue,
                                  lvalue, mlir::ValueRange{});
  }

  if (isPrefix) {
    return newValue;
  } else {
    return oldValue;
  }
}

mlir::Value
CMLIRCASTVisitor::generateUnaryOperator(clang::UnaryOperator *unOp) {
  mlir::OpBuilder &builder = context_manager_.Builder();
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
      mlir::Value zero = mlir::arith::ConstantOp::create(
          builder, builder.getUnknownLoc(), type,
          builder.getIntegerAttr(type, 0));
      return mlir::arith::SubIOp::create(builder, builder.getUnknownLoc(), zero,
                                         operand)
          .getResult();
    } else if (mlir::isa<mlir::FloatType>(type)) {
      return mlir::arith::NegFOp::create(builder, builder.getUnknownLoc(),
                                         operand);
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
      mlir::Value zero = mlir::arith::ConstantOp::create(
          builder, builder.getUnknownLoc(), type,
          builder.getIntegerAttr(type, 0));
      return mlir::arith::CmpIOp::create(builder, builder.getUnknownLoc(),
                                         mlir::arith::CmpIPredicate::eq,
                                         operand, zero)
          .getResult();
    } else if (mlir::isa<mlir::FloatType>(type)) {
      mlir::Value zero = mlir::arith::ConstantOp::create(
          builder, builder.getUnknownLoc(), type,
          builder.getFloatAttr(type, 0.0));
      return mlir::arith::CmpFOp::create(builder, builder.getUnknownLoc(),
                                         mlir::arith::CmpFPredicate::OEQ,
                                         operand, zero)
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
      mlir::Value allOnes = mlir::arith::ConstantOp::create(
          builder, builder.getUnknownLoc(), type,
          builder.getIntegerAttr(type, -1));
      return mlir::arith::XOrIOp::create(builder, builder.getUnknownLoc(),
                                         operand, allOnes)
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

} // namespace cmlirc
