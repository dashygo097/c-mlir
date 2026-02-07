#include "../ASTVisitor.h"
#include "./Types.h"

namespace cmlirc {

mlir::Value CMLIRCASTVisitor::generateExpr(clang::Expr *expr, bool needLValue) {
  if (!expr)
    return nullptr;

  expr = expr->IgnoreImpCasts();

  if (auto *intLit = llvm::dyn_cast<clang::IntegerLiteral>(expr)) {
    return generateIntegerLiteral(intLit);
  }
  if (auto *floatLit = llvm::dyn_cast<clang::FloatingLiteral>(expr)) {
    return generateFloatingLiteral(floatLit);
  }
  if (auto *declRef = llvm::dyn_cast<clang::DeclRefExpr>(expr)) {
    return generateDeclRefExpr(declRef, needLValue);
  }
  if (auto *arraySubscript = llvm::dyn_cast<clang::ArraySubscriptExpr>(expr)) {
    return generateArraySubscriptExpr(arraySubscript, needLValue);
  }
  if (auto *unOp = llvm::dyn_cast<clang::UnaryOperator>(expr)) {
    return generateUnaryOperator(unOp);
  }
  if (auto *binOp = llvm::dyn_cast<clang::BinaryOperator>(expr)) {
    return generateBinaryOperator(binOp);
  }
  if (auto *callExpr = llvm::dyn_cast<clang::CallExpr>(expr)) {
    return generateCallExpr(callExpr);
  }

  llvm::outs() << "Unsupported expression conversion for expr: "
               << expr->getStmtClassName() << "\n";
  return nullptr;
}

mlir::Value
CMLIRCASTVisitor::generateIntegerLiteral(clang::IntegerLiteral *intLit) {
  mlir::OpBuilder &builder = context_manager_.Builder();

  int64_t value = intLit->getValue().getSExtValue();
  mlir::Type type = convertType(builder, intLit->getType());

  llvm::outs() << "      Integer literal: " << value << "\n";

  return mlir::arith::ConstantOp::create(builder, builder.getUnknownLoc(), type,
                                         builder.getIntegerAttr(type, value))
      .getResult();
}

mlir::Value
CMLIRCASTVisitor::generateFloatingLiteral(clang::FloatingLiteral *floatLit) {
  mlir::OpBuilder &builder = context_manager_.Builder();

  double value = floatLit->getValue().convertToDouble();
  mlir::Type type = convertType(builder, floatLit->getType());

  llvm::outs() << "      Floating literal: " << value << "\n";

  return mlir::arith::ConstantOp::create(builder, builder.getUnknownLoc(), type,
                                         builder.getFloatAttr(type, value))
      .getResult();
}

mlir::Value CMLIRCASTVisitor::generateDeclRefExpr(clang::DeclRefExpr *declRef,
                                                  bool needLValue) {
  mlir::OpBuilder &builder = context_manager_.Builder();

  if (auto *varDecl = llvm::dyn_cast<clang::VarDecl>(declRef->getDecl())) {
    llvm::outs() << "      Variable ref: " << varDecl->getNameAsString()
                 << (needLValue ? " (lvalue)" : " (rvalue)") << "\n";

    if (auto *parmDecl = llvm::dyn_cast<clang::ParmVarDecl>(varDecl)) {
      if (paramTable.count(parmDecl)) {
        llvm::outs() << "        -> Function parameter\n";
        return paramTable[parmDecl];
      }
    }

    if (symbolTable.count(varDecl)) {
      mlir::Value memref = symbolTable[varDecl];

      if (needLValue) {
        llvm::outs() << "        -> Local variable (memref)\n";
        return memref;
      } else {
        llvm::outs() << "        -> Local variable (load)\n";
        return mlir::memref::LoadOp::create(builder, builder.getUnknownLoc(),
                                            memref)
            .getResult();
      }
    }

    llvm::outs() << "        -> ERROR: Variable not found!\n";
  }

  if (auto *funcDecl =
          llvm::dyn_cast<clang::FunctionDecl>(declRef->getDecl())) {
    llvm::outs() << "      Function ref: " << funcDecl->getNameAsString()
                 << "\n";

    if (functionTable.count(funcDecl)) {
      llvm::outs() << "        -> Function found\n";
      return functionTable[funcDecl];
    }

    llvm::outs() << "        -> ERROR: Function not found!\n";
  }

  llvm::outs() << "        -> Unsupported DeclRefExpr type: "
               << declRef->getDecl()->getDeclKindName() << "\n";
  return nullptr;
}

mlir::Value
CMLIRCASTVisitor::generateArraySubscriptExpr(clang::ArraySubscriptExpr *expr,
                                             bool needLValue) {

  llvm::outs() << "      Array subscript";

  mlir::OpBuilder &builder = context_manager_.Builder();

  clang::Expr *baseExpr = expr->getBase();
  mlir::Value base = generateExpr(baseExpr, true);

  clang::Expr *idxExpr = expr->getIdx();
  mlir::Value idx = generateExpr(idxExpr);

  if (!base || !idx) {
    llvm::outs() << "\n        ERROR: Failed to generate base or index\n";
    return nullptr;
  }

  auto indexType = builder.getIndexType();
  auto indexValue = mlir::arith::IndexCastOp::create(
      builder, builder.getUnknownLoc(), indexType, idx);

  if (needLValue) {
    llvm::outs() << " (lvalue)\n";
    return base;
  } else {
    llvm::outs() << " (rvalue - load)\n";
    auto loadOp =
        mlir::memref::LoadOp::create(builder, builder.getUnknownLoc(), base,
                                     mlir::ValueRange{indexValue.getResult()});
    return loadOp.getResult();
  }
}

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
