#include "../../ArgumentList.h"
#include "../ASTVisitor.h"
#include "./Types.h"

namespace cmlirc {

mlir::Value CMLIRCASTVisitor::generateExpr(clang::Expr *expr, bool needLValue) {
  if (!expr)
    return nullptr;

  expr = expr->IgnoreImpCasts();

  if (auto parenExpr = llvm::dyn_cast<clang::ParenExpr>(expr)) {
    return generateExpr(parenExpr->getSubExpr(), needLValue);
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

mlir::Value CMLIRCASTVisitor::generateDeclRefExpr(clang::DeclRefExpr *declRef,
                                                  bool needLValue) {
  mlir::OpBuilder &builder = context_manager_.Builder();

  if (auto *varDecl = llvm::dyn_cast<clang::VarDecl>(declRef->getDecl())) {
    if (auto *parmDecl = llvm::dyn_cast<clang::ParmVarDecl>(varDecl)) {
      if (paramTable.count(parmDecl)) {
        return paramTable[parmDecl];
      }
    }

    if (symbolTable.count(varDecl)) {
      mlir::Value memref = symbolTable[varDecl];
      return needLValue ? memref
                        : mlir::memref::LoadOp::create(
                              builder, builder.getUnknownLoc(), memref)
                              .getResult();
    }

    llvm::errs() << "Variable not found!\n";
  }

  if (auto *funcDecl =
          llvm::dyn_cast<clang::FunctionDecl>(declRef->getDecl())) {

    if (functionTable.count(funcDecl)) {
      return functionTable[funcDecl];
    }

    llvm::errs() << "Function not found!\n";
  }

  llvm::errs() << "Unsupported DeclRefExpr type: "
               << declRef->getDecl()->getDeclKindName() << "\n";
  return nullptr;
}

mlir::Value
CMLIRCASTVisitor::generateArraySubscriptExpr(clang::ArraySubscriptExpr *expr,
                                             bool needLValue) {

  mlir::OpBuilder &builder = context_manager_.Builder();

  mlir::Value base = generateExpr(expr->getBase(), true);
  mlir::Value idx = generateExpr(expr->getIdx());

  if (!base || !idx) {
    llvm::errs() << "Failed to generate base or index\n ";
    return nullptr;
  }

  auto indexValue = mlir::arith::IndexCastOp::create(
      builder, builder.getUnknownLoc(), builder.getIndexType(), idx);

  if (needLValue) {
    return base;
  } else {
    auto loadOp =
        mlir::memref::LoadOp::create(builder, builder.getUnknownLoc(), base,
                                     mlir::ValueRange{indexValue.getResult()});
    return loadOp.getResult();
  }
}

} // namespace cmlirc
