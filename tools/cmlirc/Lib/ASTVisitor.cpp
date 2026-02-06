#include "./ASTVisitor.h"
#include "./Conversions/Types.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "clang/AST/Expr.h"
#include "clang/AST/Stmt.h"
#include "clang/Basic/SourceManager.h"

namespace cmlirc {
using namespace clang;

CMLIRCASTVisitor::CMLIRCASTVisitor(ContextManager &ctx)
    : context_manager_(ctx) {}

bool CMLIRCASTVisitor::TraverseFunctionDecl(clang::FunctionDecl *decl) {
  if (decl->isImplicit() || !decl->hasBody()) {
    return true;
  }

  llvm::outs() << "\n=== Processing Function: " << decl->getNameAsString()
               << " ===\n";

  mlir::OpBuilder &builder = context_manager_.Builder();
  builder.setInsertionPointToEnd(context_manager_.Module().getBody());

  // Convert parameter types
  llvm::SmallVector<mlir::Type, 4> argTypes;
  for (auto *param : decl->parameters()) {
    mlir::Type paramType = convertType(builder, param->getType());
    argTypes.push_back(paramType);
    llvm::outs() << "  Parameter: " << param->getNameAsString() << " : "
                 << param->getType().getAsString() << "\n";
  }

  // Convert return type
  mlir::Type returnType = convertType(builder, decl->getReturnType());
  auto funcType = builder.getFunctionType(argTypes, {returnType});

  // Create function
  auto funcOp = mlir::func::FuncOp::create(builder, builder.getUnknownLoc(),
                                           decl->getNameAsString(), funcType);

  // Create entry block
  auto *entryBlock = funcOp.addEntryBlock();
  builder.setInsertionPointToStart(entryBlock);

  currentFunc = funcOp;

  // Map parameters to block arguments
  for (unsigned i = 0; i < decl->getNumParams(); ++i) {
    auto *param = decl->getParamDecl(i);
    paramTable[param] = entryBlock->getArgument(i);
    llvm::outs() << "  Mapped param: " << param->getNameAsString() << " -> %arg"
                 << i << "\n";
  }

  // Traverse function body manually
  llvm::outs() << "  Processing function body...\n";
  TraverseStmt(decl->getBody());

  llvm::outs() << "=== Finished Function ===\n\n";

  // Return true, but we've already handled traversal manually
  return true;
}

bool CMLIRCASTVisitor::VisitVarDecl(VarDecl *decl) {
  if (decl->isImplicit()) {
    return true;
  }

  if (llvm::isa<clang::ParmVarDecl>(decl)) {
    llvm::outs() << "  Skipping parameter VarDecl: " << decl->getNameAsString()
                 << "\n";
    return true;
  }

  if (!currentFunc) {
    return true;
  }

  llvm::outs() << "  Local variable: " << decl->getNameAsString() << " : "
               << decl->getType().getAsString() << "\n";

  SourceManager &SM = context_manager_.ClangContext().getSourceManager();
  SourceLocation loc = decl->getLocation();

  auto mlirLoc = mlir::FileLineColLoc::get(
      &context_manager_.MLIRContext(), SM.getFilename(loc),
      SM.getSpellingLineNumber(loc), SM.getSpellingColumnNumber(loc));

  mlir::OpBuilder &builder = context_manager_.Builder();

  QualType clangType = decl->getType();
  mlir::Type mlirType = convertType(builder, clangType);

  // Create alloca
  auto allocaOp = mlir::memref::AllocaOp::create(
      builder, mlirLoc, mlir::MemRefType::get({}, mlirType));

  symbolTable[decl] = allocaOp.getResult();

  if (decl->hasInit()) {
    Expr *init = decl->getInit();
    mlir::Value initValue = generateExpr(init);

    if (initValue) {
      mlir::memref::StoreOp::create(builder, mlirLoc, initValue,
                                    allocaOp.getResult(), mlir::ValueRange{});
      llvm::outs() << "    Initialized\n";
    }
  }

  return true;
}

bool CMLIRCASTVisitor::VisitReturnStmt(clang::ReturnStmt *stmt) {
  if (!currentFunc) {
    return true;
  }

  llvm::outs() << "  Return statement\n";

  mlir::OpBuilder &builder = context_manager_.Builder();

  mlir::Value retValue = nullptr;
  if (auto *retExpr = stmt->getRetValue()) {
    llvm::outs() << "    Generating return expression...\n";
    retValue = generateExpr(retExpr);
  }

  if (retValue) {
    mlir::func::ReturnOp::create(builder, builder.getUnknownLoc(),
                                 mlir::ValueRange{retValue});
    llvm::outs() << "    Generated: func.return with value\n";
  } else {
    mlir::func::ReturnOp::create(builder, builder.getUnknownLoc());
    llvm::outs() << "    Generated: func.return (void)\n";
  }

  return true;
}

mlir::Value CMLIRCASTVisitor::generateExpr(clang::Expr *expr) {
  if (!expr)
    return nullptr;

  mlir::OpBuilder &builder = context_manager_.Builder();
  expr = expr->IgnoreImpCasts();

  // Integer literal
  if (auto *intLit = llvm::dyn_cast<clang::IntegerLiteral>(expr)) {
    int64_t value = intLit->getValue().getSExtValue();
    mlir::Type type = convertType(builder, expr->getType());
    llvm::outs() << "      Integer literal: " << value << "\n";
    return mlir::arith::ConstantOp::create(builder, builder.getUnknownLoc(),
                                           type,
                                           builder.getIntegerAttr(type, value))
        .getResult();
  }

  // Float literal
  if (auto *floatLit = llvm::dyn_cast<clang::FloatingLiteral>(expr)) {
    llvm::APFloat value = floatLit->getValue();
    mlir::Type type = convertType(builder, expr->getType());
    llvm::outs() << "      Float literal\n";
    return mlir::arith::ConstantOp::create(builder, builder.getUnknownLoc(),
                                           type,
                                           builder.getFloatAttr(type, value))
        .getResult();
  }

  // Variable reference
  if (auto *declRef = llvm::dyn_cast<clang::DeclRefExpr>(expr)) {
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
  }

  // Binary operator
  if (auto *binOp = llvm::dyn_cast<clang::BinaryOperator>(expr)) {
    llvm::outs() << "      Binary operator: " << binOp->getOpcodeStr().str()
                 << "\n";

    mlir::Value lhs = generateExpr(binOp->getLHS());
    mlir::Value rhs = generateExpr(binOp->getRHS());

    if (!lhs || !rhs) {
      llvm::outs() << "        ERROR: Failed to generate operands\n";
      return nullptr;
    }

    switch (binOp->getOpcode()) {
    case clang::BO_Add:
      llvm::outs() << "        -> arith.addi\n";
      return mlir::arith::AddIOp::create(builder, builder.getUnknownLoc(), lhs,
                                         rhs)
          .getResult();

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
      llvm::outs() << "        Unsupported operator\n";
      return nullptr;
    }
  }

  llvm::outs() << "      Unsupported expression\n";
  expr->dump();
  return nullptr;
}

} // namespace cmlirc
