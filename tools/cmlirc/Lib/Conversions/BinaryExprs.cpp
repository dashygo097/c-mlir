#include "../../ArgumentList.h"
#include "../ASTVisitor.h"
#include "./Types.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "clang/AST/OperationKinds.h"

namespace cmlirc {

mlir::Value
CMLIRCASTVisitor::generateBinaryOperator(clang::BinaryOperator *binOp) {
  // TODO: signed/unsigned distinction for integer operations
#define REGISTER_BIN_IOP(op, ...)                                              \
  if (mlir::isa<mlir::IntegerType>(resultType)) {                              \
    return mlir::arith::op::create(builder, builder.getUnknownLoc(),           \
                                   __VA_ARGS__)                                \
        .getResult();                                                          \
  }

#define REGISTER_BIN_FOP(op, ...)                                              \
  if (mlir::isa<mlir::FloatType>(resultType)) {                                \
    return mlir::arith::op::create(builder, builder.getUnknownLoc(),           \
                                   __VA_ARGS__)                                \
        .getResult();                                                          \
  }

  mlir::OpBuilder &builder = context_manager_.Builder();

  if (binOp->getOpcode() == clang::BO_LAnd) {
    return generateLogicalAnd(binOp);
  }
  if (binOp->getOpcode() == clang::BO_LOr) {
    return generateLogicalOr(binOp);
  }

  clang::Expr *lhs = binOp->getLHS();
  clang::Expr *rhs = binOp->getRHS();
  mlir::Value lhsValue = generateExpr(lhs);
  mlir::Value rhsValue = generateExpr(rhs);

  mlir::Type resultType = lhsValue.getType();

  if (!lhs || !rhs) {
    llvm::errs() << "Failed to generate LHS or RHS\n";
    return nullptr;
  }

  switch (binOp->getOpcode()) {
  case clang::BO_Assign: {
    mlir::Value lhsBase = generateExpr(lhs, /*needLValue=*/true);
    if (!lhsBase) {
      llvm::errs() << "Failed to generate LHS\n";
      return nullptr;
    }

    if (llvm::isa<clang::ArraySubscriptExpr>(lhs)) {
      if (!lastArrayAccess_) {
        llvm::errs() << "Error: Array access info not saved\n";
        return nullptr;
      }

      mlir::memref::StoreOp::create(builder, builder.getUnknownLoc(), rhsValue,
                                    lastArrayAccess_->base,
                                    lastArrayAccess_->indices);
      lastArrayAccess_.reset();
    } else {
      mlir::memref::StoreOp::create(builder, builder.getUnknownLoc(), rhsValue,
                                    lhsBase, mlir::ValueRange{});
    }

    return rhsValue;
    break;
  }

  case clang::BO_Add: {
    REGISTER_BIN_IOP(AddIOp, lhsValue, rhsValue)
    REGISTER_BIN_FOP(AddFOp, lhsValue, rhsValue)
    break;
  }
  case clang::BO_Sub: {
    REGISTER_BIN_IOP(SubIOp, lhsValue, rhsValue)
    REGISTER_BIN_FOP(SubFOp, lhsValue, rhsValue)
    break;
  }
  case clang::BO_Mul: {
    REGISTER_BIN_IOP(MulIOp, lhsValue, rhsValue)
    REGISTER_BIN_FOP(MulFOp, lhsValue, rhsValue)
    break;
  }
  case clang::BO_Div: {
    REGISTER_BIN_IOP(DivSIOp, lhsValue, rhsValue)
    REGISTER_BIN_FOP(DivFOp, lhsValue, rhsValue)
    break;
  }
  case clang::BO_Rem: {
    REGISTER_BIN_IOP(RemSIOp, lhsValue, rhsValue)
    break;
  }
  case clang::BO_And: {
    REGISTER_BIN_IOP(AndIOp, lhsValue, rhsValue)
    break;
  }
  case clang::BO_Or: {
    REGISTER_BIN_IOP(OrIOp, lhsValue, rhsValue)
    break;
  }
  case clang::BO_Xor: {
    REGISTER_BIN_IOP(XOrIOp, lhsValue, rhsValue)
    break;
  }
  case clang::BO_Shl: {
    REGISTER_BIN_IOP(ShLIOp, lhsValue, rhsValue)
    break;
  }
  case clang::BO_Shr: {
    REGISTER_BIN_IOP(ShRSIOp, lhsValue, rhsValue)
    break;
  }
  case clang::BO_LT: {
    REGISTER_BIN_IOP(CmpIOp, mlir::arith::CmpIPredicate::slt, lhsValue,
                     rhsValue)
    REGISTER_BIN_FOP(CmpFOp, mlir::arith::CmpFPredicate::OLT, lhsValue,
                     rhsValue)
    break;
  }
  case clang::BO_LE: {
    REGISTER_BIN_IOP(CmpIOp, mlir::arith::CmpIPredicate::sle, lhsValue,
                     rhsValue)
    REGISTER_BIN_FOP(CmpFOp, mlir::arith::CmpFPredicate::OLE, lhsValue,
                     rhsValue)
    break;
  }
  case clang::BO_GT: {
    REGISTER_BIN_IOP(CmpIOp, mlir::arith::CmpIPredicate::sgt, lhsValue,
                     rhsValue)
    REGISTER_BIN_FOP(CmpFOp, mlir::arith::CmpFPredicate::OGT, lhsValue,
                     rhsValue)
    break;
  }
  case clang::BO_GE: {
    REGISTER_BIN_IOP(CmpIOp, mlir::arith::CmpIPredicate::sge, lhsValue,
                     rhsValue)
    REGISTER_BIN_FOP(CmpFOp, mlir::arith::CmpFPredicate::OGE, lhsValue,
                     rhsValue)
    break;
  }
  case clang::BO_EQ: {
    REGISTER_BIN_IOP(CmpIOp, mlir::arith::CmpIPredicate::eq, lhsValue, rhsValue)
    REGISTER_BIN_FOP(CmpFOp, mlir::arith::CmpFPredicate::OEQ, lhsValue,
                     rhsValue)
    break;
  }
  case clang::BO_NE: {
    REGISTER_BIN_IOP(CmpIOp, mlir::arith::CmpIPredicate::ne, lhsValue, rhsValue)
    REGISTER_BIN_FOP(CmpFOp, mlir::arith::CmpFPredicate::ONE, lhsValue,
                     rhsValue)
    break;
  }

  default:
    llvm::errs() << "Unsupported binary operator: "
                 << clang::BinaryOperator::getOpcodeStr(binOp->getOpcode())
                 << "\n";
  }

#undef REGISTER_BIN_IOP
#undef REGISTER_BIN_FOP

  return nullptr;
}

mlir::Value CMLIRCASTVisitor::generateLogicalAnd(clang::BinaryOperator *binOp) {
  mlir::OpBuilder &builder = context_manager_.Builder();

  mlir::Value lhs = generateExpr(binOp->getLHS());
  if (!lhs)
    return nullptr;

  mlir::Value lhsCond = convertToBool(lhs);

  auto ifOp = mlir::scf::IfOp::create(builder, builder.getUnknownLoc(),
                                      /*resultTypes=*/builder.getI1Type(),
                                      /*cond=*/lhsCond,
                                      /*withElseRegion=*/true);

  builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
  mlir::Value rhs = generateExpr(binOp->getRHS());
  if (!rhs)
    return nullptr;
  mlir::Value rhsCond = convertToBool(rhs);
  mlir::scf::YieldOp::create(builder, builder.getUnknownLoc(), rhsCond);

  builder.setInsertionPointToStart(&ifOp.getElseRegion().front());
  mlir::Value falseBool = mlir::arith::ConstantOp::create(
      builder, builder.getUnknownLoc(), builder.getI1Type(),
      builder.getBoolAttr(false));
  mlir::scf::YieldOp::create(builder, builder.getUnknownLoc(), falseBool);

  builder.setInsertionPointAfter(ifOp);
  return ifOp.getResult(0);
}

mlir::Value CMLIRCASTVisitor::generateLogicalOr(clang::BinaryOperator *binOp) {
  mlir::OpBuilder &builder = context_manager_.Builder();

  mlir::Value lhs = generateExpr(binOp->getLHS());
  if (!lhs)
    return nullptr;

  mlir::Value lhsCond = convertToBool(lhs);

  auto ifOp = mlir::scf::IfOp::create(builder, builder.getUnknownLoc(),
                                      /*resultTypes=*/builder.getI1Type(),
                                      /*cond=*/lhsCond,
                                      /*withElseRegion=*/true);

  builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
  mlir::Value trueBool = mlir::arith::ConstantOp::create(
      builder, builder.getUnknownLoc(), builder.getI1Type(),
      builder.getBoolAttr(true));
  mlir::scf::YieldOp::create(builder, builder.getUnknownLoc(), trueBool);

  builder.setInsertionPointToStart(&ifOp.getElseRegion().front());
  mlir::Value rhs = generateExpr(binOp->getRHS());
  if (!rhs)
    return nullptr;
  mlir::Value rhsCond = convertToBool(rhs);
  mlir::scf::YieldOp::create(builder, builder.getUnknownLoc(), rhsCond);

  builder.setInsertionPointAfter(ifOp);
  return ifOp.getResult(0);
}

mlir::Value CMLIRCASTVisitor::convertToBool(mlir::Value value) {
  mlir::OpBuilder &builder = context_manager_.Builder();
  mlir::Type type = value.getType();

  if (type.isInteger(1)) {
    return value;
  }

  if (auto intType = mlir::dyn_cast<mlir::IntegerType>(type)) {
    mlir::Value zero =
        mlir::arith::ConstantOp::create(builder, builder.getUnknownLoc(), type,
                                        builder.getIntegerAttr(type, 0));
    return mlir::arith::CmpIOp::create(builder, builder.getUnknownLoc(),
                                       mlir::arith::CmpIPredicate::ne, value,
                                       zero)
        .getResult();
  }

  if (auto floatType = mlir::dyn_cast<mlir::FloatType>(type)) {
    mlir::Value zero =
        mlir::arith::ConstantOp::create(builder, builder.getUnknownLoc(), type,
                                        builder.getFloatAttr(type, 0.0));
    return mlir::arith::CmpFOp::create(
               builder, builder.getUnknownLoc(),
               mlir::arith::CmpFPredicate::ONE, // ONE = Ordered Not Equal
               value, zero)
        .getResult();
  }

  llvm::errs() << "Cannot convert type to bool\n";
  return nullptr;
}

} // namespace cmlirc
