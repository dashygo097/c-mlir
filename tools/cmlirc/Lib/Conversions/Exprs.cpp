#include "../ASTVisitor.h"
#include "./Types.h"

namespace cmlirc {

mlir::Value CMLIRCASTVisitor::generateExpr(clang::Expr *expr) {
  if (!expr)
    return nullptr;

  expr = expr->IgnoreImpCasts();

  if (auto *intLit = llvm::dyn_cast<clang::IntegerLiteral>(expr)) {
    return generateIntegerLiteral(intLit);
  } else if (auto *floatLit = llvm::dyn_cast<clang::FloatingLiteral>(expr)) {
    return generateFloatingLiteral(floatLit);
  } else if (auto *declRef = llvm::dyn_cast<clang::DeclRefExpr>(expr)) {
    return generateDeclRefExpr(declRef);
  } else if (auto *unOp = llvm::dyn_cast<clang::UnaryOperator>(expr)) {
    return generateUnaryOperator(unOp);
  } else if (auto *binOp = llvm::dyn_cast<clang::BinaryOperator>(expr)) {
    return generateBinaryOperator(binOp);
  }

  llvm::outs() << "      Unsupported expression\n";
  expr->dump();
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

mlir::Value CMLIRCASTVisitor::generateDeclRefExpr(clang::DeclRefExpr *declRef) {
  mlir::OpBuilder &builder = context_manager_.Builder();

  if (auto *varDecl = llvm::dyn_cast<clang::VarDecl>(declRef->getDecl())) {
    llvm::outs() << "      Variable ref: " << varDecl->getNameAsString()
                 << "\n";

    if (auto *parmDecl = llvm::dyn_cast<clang::ParmVarDecl>(varDecl)) {
      if (paramTable.count(parmDecl)) {
        llvm::outs() << "        -> Function parameter\n";
        return paramTable[parmDecl];
      }
    }

    if (symbolTable.count(varDecl)) {
      llvm::outs() << "        -> Local variable (load)\n";
      mlir::Value memref = symbolTable[varDecl];
      return mlir::memref::LoadOp::create(builder, builder.getUnknownLoc(),
                                          memref)
          .getResult();
    }

    llvm::outs() << "        -> ERROR: Variable not found!\n";
  }

  return nullptr;
}

mlir::Value
CMLIRCASTVisitor::generateUnaryOperator(clang::UnaryOperator *unOp) {
  llvm::outs() << "      Unary operator: "
               << clang::UnaryOperator::getOpcodeStr(unOp->getOpcode()) << "\n";

  mlir::OpBuilder &builder = context_manager_.Builder();

  clang::Expr *subExpr = unOp->getSubExpr();
  mlir::Value subValue = generateExpr(subExpr);

  if (!subValue)
    return nullptr;

  switch (unOp->getOpcode()) {
  case clang::UO_Minus: {
    return mlir::arith::NegFOp::create(builder, builder.getUnknownLoc(),
                                       subValue);
  }
  default:
    llvm::outs() << "Unsupported unary operator: "
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

  mlir::Value lhs = generateExpr(binOp->getLHS());
  mlir::Value rhs = generateExpr(binOp->getRHS());

  if (!lhs || !rhs) {
    llvm::outs() << "        ERROR: Failed to generate LHS or RHS\n";
    return nullptr;
  }

  switch (binOp->getOpcode()) {
  case clang::BO_Add: {
    return mlir::arith::AddIOp::create(builder, builder.getUnknownLoc(), lhs,
                                       rhs);
  }
  case clang::BO_Sub:
    return mlir::arith::SubIOp::create(builder, builder.getUnknownLoc(), lhs,
                                       rhs)
        .getResult();
  case clang::BO_Mul:
    return mlir::arith::MulIOp::create(builder, builder.getUnknownLoc(), lhs,
                                       rhs)
        .getResult();
  case clang::BO_Div:
    return mlir::arith::DivSIOp::create(builder, builder.getUnknownLoc(), lhs,
                                        rhs)
        .getResult();
  default:
    llvm::outs() << "Unsupported binary operator: "
                 << clang::BinaryOperator::getOpcodeStr(binOp->getOpcode())
                 << "\n";
    return nullptr;
  }
}

} // namespace cmlirc
