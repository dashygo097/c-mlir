#include "../../Converter.h"
#include "../Utils/Constants.h"
#include "../Utils/Operators.h"
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
    computeResult =                                                            \
        mlir::arith::op::create(builder, loc, __VA_ARGS__).getResult();        \
  }

#define REGISTER_BIN_FOP(op, ...)                                              \
  if (mlir::isa<mlir::FloatType>(resultType)) {                                \
    return mlir::arith::op::create(builder, loc, __VA_ARGS__).getResult();     \
  }

#define REGISTER_ASSIGN_FOP(op, ...)                                           \
  if (mlir::isa<mlir::FloatType>(valueType)) {                                 \
    computeResult =                                                            \
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
      if (lastArrayAccess) {
        savedLHSAccess = lastArrayAccess;
        lastArrayAccess.reset();
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
      } else if (isMemberLHS) {
        oldValue = mlir::LLVM::LoadOp::create(builder, loc, memberPtr.getType(),
                                              memberPtr);
      } else if (isScalarLHS) {
        oldValue =
            mlir::memref::LoadOp::create(builder, loc, lhsMemref).getResult();
      }

      mlir::Type lhsType = oldValue.getType();
      mlir::Value computeLHS = oldValue;

      if (auto compOp = mlir::dyn_cast<clang::CompoundAssignOperator>(binOp)) {
        mlir::Type computeType =
            convertType(compOp->getComputationResultType());

        // Promote oldValue to compute type if needed
        if (computeType != lhsType) {
          auto srcInt = mlir::dyn_cast<mlir::IntegerType>(lhsType);
          auto dstInt = mlir::dyn_cast<mlir::IntegerType>(computeType);
          auto srcFlt = mlir::dyn_cast<mlir::FloatType>(lhsType);
          auto dstFlt = mlir::dyn_cast<mlir::FloatType>(computeType);

          if (srcInt && dstInt) {
            bool isSigned = compOp->getLHS()->getType()->isSignedIntegerType();
            if (srcInt.getWidth() < dstInt.getWidth()) {
              computeLHS = isSigned ? mlir::arith::ExtSIOp::create(
                                          builder, loc, computeType, computeLHS)
                                          .getResult()
                                    : mlir::arith::ExtUIOp::create(
                                          builder, loc, computeType, computeLHS)
                                          .getResult();
            }
          } else if (srcFlt && dstFlt) {
            if (srcFlt.getWidth() < dstFlt.getWidth())
              computeLHS = mlir::arith::ExtFOp::create(builder, loc,
                                                       computeType, computeLHS)
                               .getResult();
          } else if (srcInt && dstFlt) {
            bool isSigned = compOp->getLHS()->getType()->isSignedIntegerType();
            computeLHS = isSigned ? mlir::arith::SIToFPOp::create(
                                        builder, loc, computeType, computeLHS)
                                        .getResult()
                                  : mlir::arith::UIToFPOp::create(
                                        builder, loc, computeType, computeLHS)
                                        .getResult();
          }
        }
      }

      mlir::Value computeResult;
      mlir::Type valueType = computeLHS.getType();

      switch (binOp->getOpcode()) {
      case clang::BO_AddAssign: {
        REGISTER_ASSIGN_IOP(AddIOp, computeLHS, rhsValue)
        REGISTER_ASSIGN_FOP(AddFOp, computeLHS, rhsValue)
        break;
      }
      case clang::BO_SubAssign: {
        REGISTER_ASSIGN_IOP(SubIOp, computeLHS, rhsValue)
        REGISTER_ASSIGN_FOP(SubFOp, computeLHS, rhsValue)
        break;
      }
      case clang::BO_MulAssign: {
        REGISTER_ASSIGN_IOP(MulIOp, computeLHS, rhsValue)
        REGISTER_ASSIGN_FOP(MulFOp, computeLHS, rhsValue)
        break;
      }
      case clang::BO_DivAssign: {
        REGISTER_ASSIGN_IOP(DivSIOp, computeLHS, rhsValue)
        REGISTER_ASSIGN_FOP(DivFOp, computeLHS, rhsValue)
        break;
      }
      case clang::BO_RemAssign: {
        REGISTER_ASSIGN_IOP(RemSIOp, computeLHS, rhsValue) break;
      }
      case clang::BO_AndAssign: {
        REGISTER_ASSIGN_IOP(AndIOp, computeLHS, rhsValue) break;
      }
      case clang::BO_OrAssign: {
        REGISTER_ASSIGN_IOP(OrIOp, computeLHS, rhsValue) break;
      }
      case clang::BO_XorAssign: {
        REGISTER_ASSIGN_IOP(XOrIOp, computeLHS, rhsValue) break;
      }
      case clang::BO_ShlAssign: {
        REGISTER_ASSIGN_IOP(ShLIOp, computeLHS, rhsValue) break;
      }
      case clang::BO_ShrAssign: {
        REGISTER_ASSIGN_IOP(ShRSIOp, computeLHS, rhsValue) break;
      }
      default:
        llvm::errs() << "Unsupported compound assignment: "
                     << clang::BinaryOperator::getOpcodeStr(binOp->getOpcode())
                     << "\n";
        return nullptr;
      }

      resultValue = computeResult;
      if (computeResult && computeResult.getType() != lhsType) {
        auto srcInt =
            mlir::dyn_cast<mlir::IntegerType>(computeResult.getType());
        auto dstInt = mlir::dyn_cast<mlir::IntegerType>(lhsType);
        auto srcFlt = mlir::dyn_cast<mlir::FloatType>(computeResult.getType());
        auto dstFlt = mlir::dyn_cast<mlir::FloatType>(lhsType);

        if (srcInt && dstInt && srcInt.getWidth() > dstInt.getWidth())
          resultValue = mlir::arith::TruncIOp::create(builder, loc, lhsType,
                                                      computeResult)
                            .getResult();
        else if (srcFlt && dstFlt && srcFlt.getWidth() > dstFlt.getWidth())
          resultValue = mlir::arith::TruncFOp::create(builder, loc, lhsType,
                                                      computeResult)
                            .getResult();
      }
    }

    if (isIndexedLHS && savedLHSAccess) {
      mlir::memref::StoreOp::create(builder, loc, resultValue,
                                    savedLHSAccess->base,
                                    savedLHSAccess->indices);
    } else if (isMemberLHS) {
      mlir::LLVM::StoreOp::create(builder, loc, resultValue, memberPtr);
    } else if (isScalarLHS) {
      mlir::memref::StoreOp::create(builder, loc, resultValue, lhsMemref);
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
    mlir::scf::YieldOp::create(builder, loc,
                               detail::boolConst(builder, loc, false));
    builder.setInsertionPointAfter(ifOp);
    return ifOp.getResult(0);
  }
  case clang::BO_LOr: {
    mlir::Value lhsCond = convertToBool(lhsValue);
    auto ifOp = mlir::scf::IfOp::create(
        builder, loc, mlir::TypeRange{builder.getI1Type()}, lhsCond,
        /*withElseRegion=*/true);
    builder.setInsertionPointToStart(&ifOp.getThenRegion().front());
    mlir::scf::YieldOp::create(builder, loc,
                               detail::boolConst(builder, loc, true));
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
