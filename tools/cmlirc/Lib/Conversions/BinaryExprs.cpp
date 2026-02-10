#include "../../ArgumentList.h"
#include "../ASTVisitor.h"
#include "./Types.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "clang/AST/OperationKinds.h"

namespace cmlirc {

mlir::Value
CMLIRCASTVisitor::generateBinaryOperator(clang::BinaryOperator *binOp) {
#define REGISTER_BIN_IOP(op, ...)                                              \
  if (mlir::isa<mlir::IntegerType>(resultType)) {                              \
    return mlir::arith::op::create(builder, loc, __VA_ARGS__).getResult();     \
  }

#define REGISTER_BIN_FOP(op, ...)                                              \
  if (mlir::isa<mlir::FloatType>(resultType)) {                                \
    return mlir::arith::op::create(builder, loc, __VA_ARGS__).getResult();     \
  }

  mlir::OpBuilder &builder = context_manager_.Builder();
  mlir::Location loc = builder.getUnknownLoc();

  clang::Expr *lhs = binOp->getLHS();
  clang::Expr *rhs = binOp->getRHS();

  if (binOp->isAssignmentOp()) {
    clang::Expr *lhsLValue = lhs->IgnoreParenImpCasts();
    mlir::Value lhsBase = generateExpr(lhsLValue);

    if (!lhsBase) {
      llvm::errs() << "Failed to generate LHS\n";
      return nullptr;
    }

    mlir::Value rhsValue = generateExpr(rhs);
    if (!rhsValue) {
      llvm::errs() << "Failed to generate RHS\n";
      return nullptr;
    }

    if (llvm::isa<clang::ArraySubscriptExpr>(lhsLValue)) {
      if (!lastArrayAccess_) {
        llvm::errs() << "Error: Array access info not saved\n";
        return nullptr;
      }

      mlir::memref::StoreOp::create(builder, loc, rhsValue,
                                    lastArrayAccess_->base,
                                    lastArrayAccess_->indices);
      lastArrayAccess_.reset();
    } else {
      mlir::memref::StoreOp::create(builder, loc, rhsValue, lhsBase,
                                    mlir::ValueRange{});
    }

    return rhsValue;
  }

  mlir::Value lhsValue = generateExpr(lhs);
  mlir::Value rhsValue = generateExpr(rhs);

  if (!lhsValue || !rhsValue) {
    llvm::errs() << "Failed to generate LHS or RHS\n";
    return nullptr;
  }

  mlir::Type resultType = lhsValue.getType();

  switch (binOp->getOpcode()) {
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
  case clang::BO_LAnd: {
    mlir::Value lhsCond = convertToBool(lhsValue);
    auto ifOp = mlir::scf::IfOp::create(builder, loc,
                                        /*resultTypes=*/builder.getI1Type(),
                                        /*cond=*/lhsCond,
                                        /*withElseRegion=*/true);

    builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
    mlir::Value rhsCond = convertToBool(rhsValue);
    mlir::scf::YieldOp::create(builder, loc, rhsCond);

    builder.setInsertionPointToStart(&ifOp.getElseRegion().front());
    mlir::Value falseBool =
        mlir::arith::ConstantOp::create(builder, loc, builder.getI1Type(),
                                        builder.getBoolAttr(false))
            .getResult();
    mlir::scf::YieldOp::create(builder, loc, falseBool);

    builder.setInsertionPointAfter(ifOp);
    return ifOp.getResult(0);
  }
  case clang::BO_LOr: {
    mlir::Value lhsCond = convertToBool(lhsValue);

    auto ifOp = mlir::scf::IfOp::create(builder, loc,
                                        /*resultTypes=*/builder.getI1Type(),
                                        /*cond=*/lhsCond,
                                        /*withElseRegion=*/true);

    builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
    mlir::Value trueBool =
        mlir::arith::ConstantOp::create(builder, loc, builder.getI1Type(),
                                        builder.getBoolAttr(true))
            .getResult();
    mlir::scf::YieldOp::create(builder, loc, trueBool);

    builder.setInsertionPointToStart(&ifOp.getElseRegion().front());
    mlir::Value rhsCond = convertToBool(rhsValue);
    mlir::scf::YieldOp::create(builder, loc, rhsCond);

    builder.setInsertionPointAfter(ifOp);
    return ifOp.getResult(0);
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

} // namespace cmlirc
