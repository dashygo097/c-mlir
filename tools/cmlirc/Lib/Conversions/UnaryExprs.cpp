#include "../ASTVisitor.h"
#include "./Types.h"

namespace cmlirc {

mlir::Value CMLIRCASTVisitor::generateIncrementDecrement(clang::Expr *expr,
                                                         bool isIncrement,
                                                         bool isPrefix) {

  mlir::OpBuilder &builder = context_manager_.Builder();

  mlir::Value lvalue = generateExpr(expr, /*needLValue=*/true);
  if (!lvalue) {
    llvm::outs()
        << "        ERROR: Cannot get lvalue for increment/decrement\n";
    return nullptr;
  }

  mlir::Value oldValue;

  if (auto *arraySubscript = llvm::dyn_cast<clang::ArraySubscriptExpr>(expr)) {
    mlir::Value base = generateExpr(arraySubscript->getBase(), true);
    mlir::Value idx = generateExpr(arraySubscript->getIdx());

    auto indexValue = mlir::arith::IndexCastOp::create(
        builder, builder.getUnknownLoc(), builder.getIndexType(), idx);

    oldValue =
        mlir::memref::LoadOp::create(builder, builder.getUnknownLoc(), base,
                                     mlir::ValueRange{indexValue.getResult()})
            .getResult();

    lvalue = base;

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
    llvm::outs() << "        ERROR: Unsupported type for increment/decrement\n";
    return nullptr;
  }

  mlir::Value newValue;
  if (mlir::isa<mlir::IntegerType>(type)) {
    if (isIncrement) {
      llvm::outs() << "        -> arith.addi\n";
      newValue = mlir::arith::AddIOp::create(builder, builder.getUnknownLoc(),
                                             oldValue, one)
                     .getResult();
    } else {
      llvm::outs() << "        -> arith.subi\n";
      newValue = mlir::arith::SubIOp::create(builder, builder.getUnknownLoc(),

                                             oldValue, one)
                     .getResult();
    }
  } else {
    if (isIncrement) {
      llvm::outs() << "        -> arith.addf\n";
      newValue = mlir::arith::AddFOp::create(builder, builder.getUnknownLoc(),
                                             oldValue, one)
                     .getResult();
    } else {
      llvm::outs() << "        -> arith.subf\n";
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
    llvm::outs() << "        -> returning new value\n";
    return newValue;
  } else {
    llvm::outs() << "        -> returning old value\n";
    return oldValue;
  }
}

mlir::Value
CMLIRCASTVisitor::generateUnaryOperator(clang::UnaryOperator *unOp) {
  llvm::outs() << "      Unary operator: "
               << clang::UnaryOperator::getOpcodeStr(unOp->getOpcode()) << "\n";

  mlir::OpBuilder &builder = context_manager_.Builder();
  clang::Expr *subExpr = unOp->getSubExpr();

  switch (unOp->getOpcode()) {

  case clang::UO_Plus:
    llvm::outs() << "        -> unary plus (identity)\n";
    return generateExpr(subExpr);

  case clang::UO_Minus: {
    llvm::outs() << "        -> unary minus (negation)\n";
    mlir::Value operand = generateExpr(subExpr);
    if (!operand)
      return nullptr;

    mlir::Type type = operand.getType();

    if (mlir::isa<mlir::IntegerType>(type)) {
      mlir::Value zero = mlir::arith::ConstantOp::create(
          builder, builder.getUnknownLoc(), type,
          builder.getIntegerAttr(type, 0));
      llvm::outs() << "        -> arith.subi(0, x)\n";
      return mlir::arith::SubIOp::create(builder, builder.getUnknownLoc(), zero,
                                         operand)
          .getResult();
    } else if (mlir::isa<mlir::FloatType>(type)) {
      llvm::outs() << "        -> arith.negf\n";
      return mlir::arith::NegFOp::create(builder, builder.getUnknownLoc(),
                                         operand);
    }

    llvm::outs() << "        ERROR: Unsupported type for negation\n";
    return nullptr;
  }

  case clang::UO_PreInc: {
    llvm::outs() << "        -> pre-increment (++i)\n";
    return generateIncrementDecrement(subExpr, /*isIncrement=*/true,
                                      /*isPrefix=*/true);
  }

  case clang::UO_PostInc: {
    llvm::outs() << "        -> post-increment (i++)\n";
    return generateIncrementDecrement(subExpr, /*isIncrement=*/true,
                                      /*isPrefix=*/false);
  }

  case clang::UO_PreDec: {
    llvm::outs() << "        -> pre-decrement (--i)\n";
    return generateIncrementDecrement(subExpr, /*isIncrement=*/false,
                                      /*isPrefix=*/true);
  }

  case clang::UO_PostDec: {
    llvm::outs() << "        -> post-decrement (i--)\n";
    return generateIncrementDecrement(subExpr, /*isIncrement=*/false,
                                      /*isPrefix=*/false);
  }

  case clang::UO_LNot: {
    llvm::outs() << "        -> logical not (!x)\n";
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
    llvm::outs() << "        -> bitwise not (~x)\n";
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

    llvm::outs() << "        ERROR: Bitwise not requires integer type\n";
    return nullptr;
  }

  default:
    llvm::outs() << "        Unsupported unary operator: "
                 << clang::UnaryOperator::getOpcodeStr(unOp->getOpcode())
                 << "\n";
    return nullptr;
  }
}

} // namespace cmlirc
