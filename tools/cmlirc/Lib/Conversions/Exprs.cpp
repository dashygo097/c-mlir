#include "../../ArgumentList.h"
#include "../ASTVisitor.h"
#include "./Types.h"

namespace cmlirc {

mlir::Value CMLIRCASTVisitor::generateExpr(clang::Expr *expr) {
  if (!expr)
    return nullptr;

  if (auto parenExpr = llvm::dyn_cast<clang::ParenExpr>(expr)) {
    return generateExpr(parenExpr->getSubExpr());
  }
  if (auto *boolLit = llvm::dyn_cast<clang::CXXBoolLiteralExpr>(expr)) {
    return generateBoolLiteral(boolLit);
  }
  if (auto *intLit = llvm::dyn_cast<clang::IntegerLiteral>(expr)) {
    return generateIntegerLiteral(intLit);
  }
  if (auto *floatLit = llvm::dyn_cast<clang::FloatingLiteral>(expr)) {
    return generateFloatingLiteral(floatLit);
  }
  if (auto *declRef = llvm::dyn_cast<clang::DeclRefExpr>(expr)) {
    return generateDeclRefExpr(declRef);
  }
  if (auto *implicitCast = llvm::dyn_cast<clang::ImplicitCastExpr>(expr)) {
    return generateImplicitCastExpr(implicitCast);
  }
  if (auto *arraySubscript = llvm::dyn_cast<clang::ArraySubscriptExpr>(expr)) {
    return generateArraySubscriptExpr(arraySubscript);
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

  llvm::errs() << "Unsupported expression conversion for expr: "
               << expr->getStmtClassName() << "\n";
  return nullptr;
}

mlir::Value
CMLIRCASTVisitor::generateBoolLiteral(clang::CXXBoolLiteralExpr *boolLit) {
  mlir::OpBuilder &builder = context_manager_.Builder();

  bool value = boolLit->getValue();
  mlir::Type type = convertType(builder, boolLit->getType());

  return mlir::arith::ConstantOp::create(
             builder, builder.getUnknownLoc(), type,
             builder.getIntegerAttr(type, value ? 1 : 0))
      .getResult();
}

mlir::Value
CMLIRCASTVisitor::generateIntegerLiteral(clang::IntegerLiteral *intLit) {
  mlir::OpBuilder &builder = context_manager_.Builder();

  int64_t value = intLit->getValue().getSExtValue();
  mlir::Type type = convertType(builder, intLit->getType());

  return mlir::arith::ConstantOp::create(builder, builder.getUnknownLoc(), type,
                                         builder.getIntegerAttr(type, value))
      .getResult();
}

mlir::Value
CMLIRCASTVisitor::generateFloatingLiteral(clang::FloatingLiteral *floatLit) {
  mlir::OpBuilder &builder = context_manager_.Builder();

  double value = floatLit->getValue().convertToDouble();
  mlir::Type type = convertType(builder, floatLit->getType());

  return mlir::arith::ConstantOp::create(builder, builder.getUnknownLoc(), type,
                                         builder.getFloatAttr(type, value))
      .getResult();
}

mlir::Value CMLIRCASTVisitor::generateDeclRefExpr(clang::DeclRefExpr *declRef) {
  if (auto *varDecl = llvm::dyn_cast<clang::VarDecl>(declRef->getDecl())) {
    if (auto *parmDecl = llvm::dyn_cast<clang::ParmVarDecl>(varDecl)) {
      if (paramTable.count(parmDecl)) {
        return paramTable[parmDecl];
      }
    }

    if (symbolTable.count(varDecl)) {
      return symbolTable[varDecl];
    }

    llvm::errs() << "Variable not found: " << varDecl->getName() << "\n";
    return nullptr;
  }

  if (auto *funcDecl =
          llvm::dyn_cast<clang::FunctionDecl>(declRef->getDecl())) {
    if (functionTable.count(funcDecl)) {
      return functionTable[funcDecl];
    }

    llvm::errs() << "Function not found: " << funcDecl->getName() << "\n";
    return nullptr;
  }

  llvm::errs() << "Unsupported DeclRefExpr type: "
               << declRef->getDecl()->getDeclKindName() << "\n";
  return nullptr;
}

mlir::Value
CMLIRCASTVisitor::generateImplicitCastExpr(clang::ImplicitCastExpr *castExpr) {
  mlir::OpBuilder &builder = context_manager_.Builder();
  mlir::Location loc = builder.getUnknownLoc();
  clang::CastKind castKind = castExpr->getCastKind();

  mlir::Value subValue = generateExpr(castExpr->getSubExpr());
  if (!subValue) {
    return nullptr;
  }

  mlir::Type targetType = convertType(builder, castExpr->getType());

  switch (castKind) {
  case clang::CK_LValueToRValue: {
    if (mlir::isa<mlir::MemRefType>(subValue.getType())) {
      if (lastArrayAccess_ && lastArrayAccess_->base == subValue) {
        mlir::Value result =
            mlir::memref::LoadOp::create(builder, loc, lastArrayAccess_->base,
                                         lastArrayAccess_->indices)
                .getResult();
        lastArrayAccess_.reset();
        return result;
      } else {
        return mlir::memref::LoadOp::create(builder, loc, subValue).getResult();
      }
    }
    return subValue;
  }

  case clang::CK_IntegralToFloating: {
    bool isSigned = castExpr->getSubExpr()->getType()->isSignedIntegerType();
    if (isSigned) {
      return mlir::arith::SIToFPOp::create(builder, loc, targetType, subValue)
          .getResult();
    } else {
      return mlir::arith::UIToFPOp::create(builder, loc, targetType, subValue)
          .getResult();
    }
  }

  case clang::CK_FloatingToIntegral: {
    bool isSigned = castExpr->getType()->isSignedIntegerType();
    if (isSigned) {
      return mlir::arith::FPToSIOp::create(builder, loc, targetType, subValue)
          .getResult();
    } else {
      return mlir::arith::FPToUIOp::create(builder, loc, targetType, subValue)
          .getResult();
    }
  }

  case clang::CK_IntegralCast: {
    auto srcIntType = mlir::dyn_cast<mlir::IntegerType>(subValue.getType());
    auto dstIntType = mlir::dyn_cast<mlir::IntegerType>(targetType);

    if (!srcIntType || !dstIntType) {
      return subValue;
    }

    unsigned srcWidth = srcIntType.getWidth();
    unsigned dstWidth = dstIntType.getWidth();

    if (srcWidth < dstWidth) {
      bool isSigned = castExpr->getSubExpr()->getType()->isSignedIntegerType();
      if (isSigned) {
        return mlir::arith::ExtSIOp::create(builder, loc, targetType, subValue)
            .getResult();
      } else {
        return mlir::arith::ExtUIOp::create(builder, loc, targetType, subValue)
            .getResult();
      }
    } else if (srcWidth > dstWidth) {
      return mlir::arith::TruncIOp::create(builder, loc, targetType, subValue)
          .getResult();
    }

    return subValue;
  }

  case clang::CK_FloatingCast: {
    auto srcFloatType = mlir::dyn_cast<mlir::FloatType>(subValue.getType());
    auto dstFloatType = mlir::dyn_cast<mlir::FloatType>(targetType);

    if (!srcFloatType || !dstFloatType) {
      return subValue;
    }

    if (srcFloatType.getWidth() < dstFloatType.getWidth()) {
      return mlir::arith::ExtFOp::create(builder, loc, targetType, subValue)
          .getResult();
    } else if (srcFloatType.getWidth() > dstFloatType.getWidth()) {
      return mlir::arith::TruncFOp::create(builder, loc, targetType, subValue)
          .getResult();
    }

    return subValue;
  }

  case clang::CK_IntegralToBoolean: {
    auto zeroAttr = builder.getIntegerAttr(subValue.getType(), 0);
    mlir::Value zero =
        mlir::arith::ConstantOp::create(builder, loc, zeroAttr).getResult();
    return mlir::arith::CmpIOp::create(
               builder, loc, mlir::arith::CmpIPredicate::ne, subValue, zero)
        .getResult();
  }

  case clang::CK_FloatingToBoolean: {
    auto zeroAttr = builder.getFloatAttr(subValue.getType(), 0.0);
    mlir::Value zero =
        mlir::arith::ConstantOp::create(builder, loc, zeroAttr).getResult();
    return mlir::arith::CmpFOp::create(
               builder, loc, mlir::arith::CmpFPredicate::UNE, subValue, zero)
        .getResult();
  }

  case clang::CK_BooleanToSignedIntegral: {
    return mlir::arith::ExtUIOp::create(builder, loc, targetType, subValue)
        .getResult();
  }

  case clang::CK_BitCast: {
    return mlir::arith::BitcastOp::create(builder, loc, targetType, subValue)
        .getResult();
  }

  case clang::CK_NoOp:
  case clang::CK_ArrayToPointerDecay:
  case clang::CK_FunctionToPointerDecay:
    return subValue;

  default:
    llvm::errs() << "Unsupported cast kind: "
                 << clang::ImplicitCastExpr::getCastKindName(castKind) << "\n";
    return subValue;
  }
}

mlir::Value
CMLIRCASTVisitor::generateArraySubscriptExpr(clang::ArraySubscriptExpr *expr) {
  mlir::OpBuilder &builder = context_manager_.Builder();
  mlir::Location loc = builder.getUnknownLoc();

  llvm::SmallVector<mlir::Value, 4> indices;
  clang::Expr *currentExpr = expr;

  while (auto *arraySubscript =
             llvm::dyn_cast<clang::ArraySubscriptExpr>(currentExpr)) {
    mlir::Value idx = generateExpr(arraySubscript->getIdx());
    if (!idx) {
      llvm::errs() << "Failed to generate index\n";
      return nullptr;
    }

    indices.insert(indices.begin(), idx);
    currentExpr = arraySubscript->getBase()->IgnoreImpCasts();
  }

  mlir::Value base = generateExpr(currentExpr);
  if (!base) {
    llvm::errs() << "Failed to generate base\n";
    return nullptr;
  }

  llvm::SmallVector<mlir::Value, 4> indexValues;
  for (mlir::Value idx : indices) {
    auto indexValue = mlir::arith::IndexCastOp::create(
                          builder, loc, builder.getIndexType(), idx)
                          .getResult();
    indexValues.push_back(indexValue);
  }

  lastArrayAccess_ = ArrayAccessInfo{base, indexValues};

  return base;
}

mlir::Value CMLIRCASTVisitor::generateCallExpr(clang::CallExpr *callExpr) {
  mlir::OpBuilder &builder = context_manager_.Builder();
  mlir::Location loc = builder.getUnknownLoc();

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

  auto callOp = mlir::func::CallOp::create(builder, loc, calleeName,
                                           mlir::TypeRange{returnTypes},
                                           mlir::ValueRange{argValues});

  if (callOp.getNumResults() > 0) {
    return callOp.getResult(0);
  }

  return nullptr;
}

} // namespace cmlirc
