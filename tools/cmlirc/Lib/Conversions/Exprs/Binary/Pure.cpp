#include "../../../Converter.h"
#include "../../Utils/Casts.h"
#include "../../Utils/Constants.h"
#include "../../Utils/Operators.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "clang/AST/OperationKinds.h"
#include "llvm/Support/WithColor.h"

namespace cmlirc {
// Handle pure value-producing binary expressions (no side effects on LHS).
auto CMLIRConverter::generatePureBinaryOperator(clang::BinaryOperator *binOp)
    -> mlir::Value {
  mlir::OpBuilder &builder = contextManager.Builder();
  mlir::Location loc = builder.getUnknownLoc();

  mlir::Value lhs = generateExpr(binOp->getLHS());
  mlir::Value rhs = generateExpr(binOp->getRHS());
  if (!lhs || !rhs) {
    llvm::WithColor::error() << "cmlirc: failed to generate binary operands\n";
    return nullptr;
  }

  using CBO = clang::BinaryOperatorKind;
  using IP = mlir::arith::CmpIPredicate;
  using FP = mlir::arith::CmpFPredicate;

  switch (binOp->getOpcode()) {
  // Arithmetic
  case CBO::BO_Add:
    return utils::emitOp<mlir::arith::AddIOp, mlir::arith::AddFOp>(builder, loc,
                                                                   lhs, rhs);
  case CBO::BO_Sub:
    return utils::emitOp<mlir::arith::SubIOp, mlir::arith::SubFOp>(builder, loc,
                                                                   lhs, rhs);
  case CBO::BO_Mul:
    return utils::emitOp<mlir::arith::MulIOp, mlir::arith::MulFOp>(builder, loc,
                                                                   lhs, rhs);
  case CBO::BO_Div:
    return utils::emitOp<mlir::arith::DivSIOp, mlir::arith::DivFOp>(
        builder, loc, lhs, rhs);
  case CBO::BO_Rem:
    return utils::emitIntOp<mlir::arith::RemSIOp>(builder, loc, lhs, rhs);

  // Bitwise
  case CBO::BO_And:
    return utils::emitIntOp<mlir::arith::AndIOp>(builder, loc, lhs, rhs);
  case CBO::BO_Or:
    return utils::emitIntOp<mlir::arith::OrIOp>(builder, loc, lhs, rhs);
  case CBO::BO_Xor:
    return utils::emitIntOp<mlir::arith::XOrIOp>(builder, loc, lhs, rhs);
  case CBO::BO_Shl:
    return utils::emitIntOp<mlir::arith::ShLIOp>(builder, loc, lhs, rhs);
  case CBO::BO_Shr:
    return utils::emitIntOp<mlir::arith::ShRSIOp>(builder, loc, lhs, rhs);

  // Cmp
  case CBO::BO_LT:
    return utils::emitCmpOp(builder, loc, IP::slt, FP::OLT, lhs, rhs);
  case CBO::BO_LE:
    return utils::emitCmpOp(builder, loc, IP::sle, FP::OLE, lhs, rhs);
  case CBO::BO_GT:
    return utils::emitCmpOp(builder, loc, IP::sgt, FP::OGT, lhs, rhs);
  case CBO::BO_GE:
    return utils::emitCmpOp(builder, loc, IP::sge, FP::OGE, lhs, rhs);
  case CBO::BO_EQ:
    return utils::emitCmpOp(builder, loc, IP::eq, FP::OEQ, lhs, rhs);
  case CBO::BO_NE:
    return utils::emitCmpOp(builder, loc, IP::ne, FP::ONE, lhs, rhs);

  // Short-circuit logical ops
  case CBO::BO_LAnd:
    return generateLAndBinaryOperator(lhs, rhs);
  case CBO::BO_LOr:
    return generateLOrBinaryOperator(lhs, rhs);

  default:
    llvm::WithColor::error()
        << "cmlirc: unsupported binary operator: "
        << clang::BinaryOperator::getOpcodeStr(binOp->getOpcode()) << "\n";
    return nullptr;
  }
}

// Short-circuit helpers
auto CMLIRConverter::generateLAndBinaryOperator(mlir::Value lhs,
                                                mlir::Value rhs)
    -> mlir::Value {
  mlir::OpBuilder &builder = contextManager.Builder();
  mlir::Location loc = builder.getUnknownLoc();

  // if (lhs) { yield rhs } else { yield false }
  auto ifOp = mlir::scf::IfOp::create(builder, loc, builder.getI1Type(),
                                      utils::toBool(builder, loc, lhs), true);
  builder.setInsertionPointToStart(&ifOp.getThenRegion().front());

  mlir::scf::YieldOp::create(builder, loc, utils::toBool(builder, loc, rhs));
  builder.setInsertionPointToStart(&ifOp.getElseRegion().front());

  mlir::scf::YieldOp::create(builder, loc,
                             utils::boolConst(builder, loc, false));
  builder.setInsertionPointAfter(ifOp);

  return ifOp.getResult(0);
}

auto CMLIRConverter::generateLOrBinaryOperator(mlir::Value lhs, mlir::Value rhs)
    -> mlir::Value {
  mlir::OpBuilder &builder = contextManager.Builder();
  mlir::Location loc = builder.getUnknownLoc();

  // if (lhs) { yield true } else { yield rhs }
  auto ifOp = mlir::scf::IfOp::create(builder, loc, builder.getI1Type(),
                                      utils::toBool(builder, loc, lhs), true);
  builder.setInsertionPointToStart(&ifOp.getThenRegion().front());

  mlir::scf::YieldOp::create(builder, loc,
                             utils::boolConst(builder, loc, true));
  builder.setInsertionPointToStart(&ifOp.getElseRegion().front());

  mlir::scf::YieldOp::create(builder, loc, utils::toBool(builder, loc, rhs));
  builder.setInsertionPointAfter(ifOp);

  return ifOp.getResult(0);
}

} // namespace cmlirc
