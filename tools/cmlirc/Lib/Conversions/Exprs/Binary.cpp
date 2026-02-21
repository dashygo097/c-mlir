#include "../../../ArgumentList.h"
#include "../../Converter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "clang/AST/OperationKinds.h"

namespace cmlirc {

mlir::Value
CMLIRConverter::generateBinaryOperator(clang::BinaryOperator *binOp) {
#define REGISTER_BIN_IOP(op, ...)                                              \
  if (mlir::isa<mlir::IntegerType>(resultType)) {                              \
    return mlir::arith::op::create(builder, loc, __VA_ARGS__).getResult();     \
  }

#define REGISTER_ASSIGN_IOP(op, ...)                                           \
  if (mlir::isa<mlir::IntegerType>(valueType)) {                               \
    resultValue =                                                              \
        mlir::arith::op::create(builder, loc, __VA_ARGS__).getResult();        \
  }

#define REGISTER_BIN_FOP(op, ...)                                              \
  if (mlir::isa<mlir::FloatType>(resultType)) {                                \
    return mlir::arith::op::create(builder, loc, __VA_ARGS__).getResult();     \
  }

#define REGISTER_ASSIGN_FOP(op, ...)                                           \
  if (mlir::isa<mlir::FloatType>(valueType)) {                                 \
    resultValue =                                                              \
        mlir::arith::op::create(builder, loc, __VA_ARGS__).getResult();        \
  }

  mlir::OpBuilder &builder = context_manager_.Builder();
  mlir::Location loc = builder.getUnknownLoc();

  clang::Expr *lhs = binOp->getLHS();
  clang::Expr *rhs = binOp->getRHS();

  if (binOp->isAssignmentOp()) {
    clang::Expr *bareLHS = lhs->IgnoreParenImpCasts();
    bool isIndexedLHS =
        mlir::isa<clang::ArraySubscriptExpr>(bareLHS) ||
        (mlir::isa<clang::UnaryOperator>(bareLHS) &&
         llvm::cast<clang::UnaryOperator>(bareLHS)->getOpcode() ==
             clang::UO_Deref);
    bool isMemberLHS = mlir::isa<clang::MemberExpr>(bareLHS);
    bool isScalarLHS = !isIndexedLHS && !isMemberLHS;

    mlir::Value lhsMemref = generateExpr(lhs);
    if (!lhsMemref) {
      llvm::errs() << "Failed to generate LHS\n";
      return nullptr;
    }

    std::optional<ArrayAccessInfo> savedLHSAccess;
    mlir::Value memberPtr;

    if (isIndexedLHS) {
      if (lastArrayAccess_) {
        savedLHSAccess = lastArrayAccess_;
        lastArrayAccess_.reset();
      } else {
        llvm::errs() << "Failed to get array access info for LHS\n";
        return nullptr;
      }
    } else if (isMemberLHS) {
      memberPtr = lhsMemref;
    }

    mlir::Value rhsValue = generateExpr(rhs);
    if (!rhsValue) {
      llvm::errs() << "Failed to generate RHS\n";
      return nullptr;
    }

    mlir::Value resultValue = rhsValue;

    if (binOp->getOpcode() != clang::BO_Assign) {
      mlir::Value oldValue;

      if (isIndexedLHS && savedLHSAccess) {
        oldValue =
            mlir::memref::LoadOp::create(builder, loc, savedLHSAccess->base,
                                         savedLHSAccess->indices)
                .getResult();
      }
      if (isMemberLHS) {
        oldValue = mlir::LLVM::LoadOp::create(builder, loc, memberPtr.getType(),
                                              memberPtr);
      }
      if (isScalarLHS) {
        oldValue =
            mlir::memref::LoadOp::create(builder, loc, lhsMemref).getResult();
      }

      mlir::Type valueType = oldValue.getType();

      switch (binOp->getOpcode()) {
      case clang::BO_AddAssign: {
        REGISTER_ASSIGN_IOP(AddIOp, oldValue, rhsValue)
        REGISTER_ASSIGN_FOP(AddFOp, oldValue, rhsValue)
        break;
      }
      case clang::BO_SubAssign: {
        REGISTER_ASSIGN_IOP(SubIOp, oldValue, rhsValue)
        REGISTER_ASSIGN_FOP(SubFOp, oldValue, rhsValue)
        break;
      }
      case clang::BO_MulAssign: {
        REGISTER_ASSIGN_IOP(MulIOp, oldValue, rhsValue)
        REGISTER_ASSIGN_FOP(MulFOp, oldValue, rhsValue)
        break;
      }
      case clang::BO_DivAssign: {
        REGISTER_ASSIGN_IOP(DivSIOp, oldValue, rhsValue)
        REGISTER_ASSIGN_FOP(DivFOp, oldValue, rhsValue)
        break;
      }
      case clang::BO_RemAssign: {
        REGISTER_ASSIGN_IOP(RemSIOp, oldValue, rhsValue)
        break;
      }
      case clang::BO_AndAssign: {
        REGISTER_ASSIGN_IOP(AndIOp, oldValue, rhsValue)
        break;
      }
      case clang::BO_OrAssign: {
        REGISTER_ASSIGN_IOP(OrIOp, oldValue, rhsValue)
        break;
      }
      case clang::BO_XorAssign: {
        REGISTER_ASSIGN_IOP(XOrIOp, oldValue, rhsValue)
        break;
      }
      case clang::BO_ShlAssign: {
        REGISTER_ASSIGN_IOP(ShLIOp, oldValue, rhsValue)
        break;
      }
      case clang::BO_ShrAssign: {
        REGISTER_ASSIGN_IOP(ShRSIOp, oldValue, rhsValue)
        break;
      }
      default:
        llvm::errs() << "Unsupported compound assignment operator: "
                     << clang::BinaryOperator::getOpcodeStr(binOp->getOpcode())
                     << "\n";
        return nullptr;
      }
    }

    if (isIndexedLHS && savedLHSAccess) {
      mlir::memref::StoreOp::create(builder, loc, resultValue,
                                    savedLHSAccess->base,
                                    savedLHSAccess->indices);
    }
    if (isMemberLHS) {
      mlir::LLVM::StoreOp::create(builder, loc, resultValue, memberPtr);
    }
    if (isScalarLHS) {
      mlir::memref::StoreOp::create(builder, loc, resultValue, lhsMemref,
                                    mlir::ValueRange{});
    }

    return resultValue;
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
    auto ifOp = mlir::scf::IfOp::create(
        builder, loc, mlir::TypeRange{builder.getI1Type()}, lhsCond,
        /*withElseRegion=*/true);
    builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
    mlir::scf::YieldOp::create(builder, loc, convertToBool(rhsValue));
    builder.setInsertionPointToStart(&ifOp.getElseRegion().front());
    mlir::scf::YieldOp::create(
        builder, loc,
        mlir::arith::ConstantOp::create(builder, loc, builder.getI1Type(),
                                        builder.getBoolAttr(false))
            .getResult());
    builder.setInsertionPointAfter(ifOp);
    return ifOp.getResult(0);
  }
  case clang::BO_LOr: {
    mlir::Value lhsCond = convertToBool(lhsValue);
    auto ifOp = mlir::scf::IfOp::create(
        builder, loc, mlir::TypeRange{builder.getI1Type()}, lhsCond,
        /*withElseRegion=*/true);
    builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
    mlir::scf::YieldOp::create(
        builder, loc,
        mlir::arith::ConstantOp::create(builder, loc, builder.getI1Type(),
                                        builder.getBoolAttr(true))
            .getResult());
    builder.setInsertionPointToStart(&ifOp.getElseRegion().front());
    mlir::scf::YieldOp::create(builder, loc, convertToBool(rhsValue));
    builder.setInsertionPointAfter(ifOp);
    return ifOp.getResult(0);
  }

  default:
    llvm::errs() << "Unsupported binary operator: "
                 << clang::BinaryOperator::getOpcodeStr(binOp->getOpcode())
                 << "\n";
  }

#undef REGISTER_BIN_IOP
#undef REGISTER_ASSIGN_IOP
#undef REGISTER_BIN_FOP
#undef REGISTER_ASSIGN_FOP

  return nullptr;
}

} // namespace cmlirc
