#include "../../ArgumentList.h"
#include "../ASTVisitor.h"
#include "./Types.h"

namespace cmlirc {

mlir::Value
CMLIRCASTVisitor::generateBinaryOperator(clang::BinaryOperator *binOp) {
  mlir::OpBuilder &builder = context_manager_.Builder();

  if (binOp->isAssignmentOp()) {
    clang::Expr *lhs = binOp->getLHS();
    clang::Expr *rhs = binOp->getRHS();

    mlir::Value rhsValue = generateExpr(rhs);
    if (!rhsValue) {
      llvm::errs() << "Failed to generate RHS\n";
      return nullptr;
    }

    if (auto *arraySubscript = llvm::dyn_cast<clang::ArraySubscriptExpr>(lhs)) {
      mlir::Value base = generateExpr(arraySubscript->getBase(), true);
      mlir::Value idx = generateExpr(arraySubscript->getIdx());

      if (!base || !idx) {
        llvm::errs() << "Failed to get base or index\n";
        return nullptr;
      }

      auto indexValue = mlir::arith::IndexCastOp::create(
          builder, builder.getUnknownLoc(), builder.getIndexType(), idx);

      mlir::memref::StoreOp::create(builder, builder.getUnknownLoc(), rhsValue,
                                    base,
                                    mlir::ValueRange{indexValue.getResult()});

      return rhsValue;

    } else {
      mlir::Value lhsMemref = generateExpr(lhs, true);
      if (!lhsMemref) {
        llvm::errs() << "Failed to generate LHS\n";
        return nullptr;
      }

      mlir::memref::StoreOp::create(builder, builder.getUnknownLoc(), rhsValue,
                                    lhsMemref, mlir::ValueRange{});

      return rhsValue;
    }
  }

  mlir::Value lhs = generateExpr(binOp->getLHS());
  mlir::Value rhs = generateExpr(binOp->getRHS());

  if (!lhs || !rhs) {
    llvm::errs() << "Failed to generate LHS or RHS\n";
    return nullptr;
  }

  mlir::Type resultType = lhs.getType();

  // TODO: signed/unsigned distinction for integer operations
#define REGISTER_BIN_IOP(op)                                                   \
  if (mlir::isa<mlir::IntegerType>(resultType)) {                              \
    return mlir::arith::op##Op::create(builder, builder.getUnknownLoc(), lhs,  \
                                       rhs)                                    \
        .getResult();                                                          \
  }

#define REGISTER_BIN_FOP(op)                                                   \
  if (mlir::isa<mlir::FloatType>(resultType)) {                                \
    return mlir::arith::op##Op::create(builder, builder.getUnknownLoc(), lhs,  \
                                       rhs)                                    \
        .getResult();                                                          \
  }

  switch (binOp->getOpcode()) {
  case clang::BO_Add:
    REGISTER_BIN_IOP(AddI)
    REGISTER_BIN_FOP(AddF)
    break;
  case clang::BO_Sub:
    REGISTER_BIN_IOP(SubI)
    REGISTER_BIN_FOP(SubF)
    break;
  case clang::BO_Mul:
    REGISTER_BIN_IOP(MulI)
    REGISTER_BIN_FOP(MulF)
    break;
  case clang::BO_Div:
    REGISTER_BIN_IOP(DivSI)
    REGISTER_BIN_FOP(DivF)
    break;
  case clang::BO_And:
    REGISTER_BIN_IOP(AndI)
    break;
  case clang::BO_Or:
    REGISTER_BIN_IOP(OrI)
    break;
  case clang::BO_Xor:
    REGISTER_BIN_IOP(XOrI)
    break;
  case clang::BO_Shl:
    REGISTER_BIN_IOP(ShLI)
    break;
  case clang::BO_Shr:
    REGISTER_BIN_IOP(ShRSI)
    break;
  default:
    llvm::errs() << "Unsupported binary operator: "
                 << clang::BinaryOperator::getOpcodeStr(binOp->getOpcode())
                 << "\n";
  }

#undef REGISTER_BIN_IOP
#undef REGISTER_BIN_FOP

  return nullptr;
}

mlir::Value CMLIRCASTVisitor::generateCallExpr(clang::CallExpr *callExpr) {
  mlir::OpBuilder &builder = context_manager_.Builder();

  const clang::FunctionDecl *calleeDecl = callExpr->getDirectCallee();
  if (!calleeDecl) {
    llvm::errs() << "Indirect calls not supported yet\n";
    return nullptr;
  }

  std::string calleeName = calleeDecl->getNameAsString();

  llvm::SmallVector<mlir::Value, 4> argValues;
  for (unsigned i = 0; i < callExpr->getNumArgs(); ++i) {
    clang::Expr *argExpr = callExpr->getArg(i);
    mlir::Value argValue = generateExpr(argExpr);
    if (!argValue) {
      llvm::errs() << "Failed to generate argument " << i << "\n";
      return nullptr;
    }
    argValues.push_back(argValue);
  }

  clang::QualType returnType = calleeDecl->getReturnType();
  mlir::Type mlirReturnType = convertType(builder, returnType);

  llvm::SmallVector<mlir::Type, 1> returnTypes;
  if (!mlir::isa<mlir::NoneType>(mlirReturnType)) {
    returnTypes.push_back(mlirReturnType);
  }

  auto callOp = mlir::func::CallOp::create(
      builder, builder.getUnknownLoc(), calleeName,
      mlir::TypeRange{returnTypes}, mlir::ValueRange{argValues});

  if (callOp.getNumResults() > 0) {
    return callOp.getResult(0);
  }

  return nullptr;
}
} // namespace cmlirc
