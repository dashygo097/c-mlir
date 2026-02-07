#include "../ASTVisitor.h"
#include "./Types.h"

namespace cmlirc {

mlir::Value
CMLIRCASTVisitor::generateBinaryOperator(clang::BinaryOperator *binOp) {
  llvm::outs() << "      Binary operator: " << binOp->getOpcodeStr().str()
               << "\n";

  mlir::OpBuilder &builder = context_manager_.Builder();

  if (binOp->isAssignmentOp()) {
    llvm::outs() << "        Assignment operation\n";

    clang::Expr *lhs = binOp->getLHS();
    clang::Expr *rhs = binOp->getRHS();

    mlir::Value rhsValue = generateExpr(rhs);
    if (!rhsValue) {
      llvm::outs() << "        ERROR: Failed to generate RHS\n";
      return nullptr;
    }

    if (auto *arraySubscript = llvm::dyn_cast<clang::ArraySubscriptExpr>(lhs)) {
      llvm::outs() << "        Assigning to array element\n";

      mlir::Value base = generateExpr(arraySubscript->getBase(), true);
      mlir::Value idx = generateExpr(arraySubscript->getIdx());

      if (!base || !idx) {
        llvm::outs() << "        ERROR: Failed to get base or index\n";
        return nullptr;
      }

      auto indexValue = mlir::arith::IndexCastOp::create(
          builder, builder.getUnknownLoc(), builder.getIndexType(), idx);

      mlir::memref::StoreOp::create(builder, builder.getUnknownLoc(), rhsValue,
                                    base,
                                    mlir::ValueRange{indexValue.getResult()});

      llvm::outs() << "        Array assignment complete\n";
      return rhsValue;

    } else {
      llvm::outs() << "        Assigning to variable\n";
      mlir::Value lhsMemref = generateExpr(lhs, true);
      if (!lhsMemref) {
        llvm::outs() << "        ERROR: Failed to generate LHS\n";
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
    llvm::outs() << "        ERROR: Failed to generate LHS or RHS\n";
    return nullptr;
  }

  mlir::Type resultType = lhs.getType();

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
  default:
    llvm::outs() << "Unsupported binary operator: "
                 << clang::BinaryOperator::getOpcodeStr(binOp->getOpcode())
                 << "\n";
  }

#undef REGISTER_BIN_IOP
#undef REGISTER_BIN_FOP

  return nullptr;
}

mlir::Value CMLIRCASTVisitor::generateCallExpr(clang::CallExpr *callExpr) {
  llvm::outs() << "      Call expression\n";

  mlir::OpBuilder &builder = context_manager_.Builder();

  const clang::FunctionDecl *calleeDecl = callExpr->getDirectCallee();
  if (!calleeDecl) {
    llvm::outs() << "        ERROR: Indirect calls not supported yet\n";
    return nullptr;
  }

  std::string calleeName = calleeDecl->getNameAsString();
  llvm::outs() << "        Calling function: " << calleeName << "\n";

  llvm::SmallVector<mlir::Value, 4> argValues;
  for (unsigned i = 0; i < callExpr->getNumArgs(); ++i) {
    clang::Expr *argExpr = callExpr->getArg(i);
    mlir::Value argValue = generateExpr(argExpr);
    if (!argValue) {
      llvm::outs() << "        ERROR: Failed to generate argument " << i
                   << "\n";
      return nullptr;
    }
    argValues.push_back(argValue);
    llvm::outs() << "        Argument " << i << " generated\n";
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
    llvm::outs() << "        Call with return value\n";
    return callOp.getResult(0);
  }

  llvm::outs() << "        Call (void)\n";
  return nullptr;
}
} // namespace cmlirc
