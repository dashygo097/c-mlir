#include "./ASTVisitor.h"
#include "../ArgumentList.h"
#include "./Conversions/Types.h"
#include "clang/Basic/SourceManager.h"

namespace cmlirc {
using namespace clang;

CMLIRCASTVisitor::CMLIRCASTVisitor(ContextManager &ctx)
    : context_manager_(ctx) {}

bool CMLIRCASTVisitor::TraverseFunctionDecl(clang::FunctionDecl *decl) {
  if (decl->isImplicit() || !decl->hasBody()) {
    return true;
  }

  mlir::OpBuilder &builder = context_manager_.Builder();
  builder.setInsertionPointToEnd(context_manager_.Module().getBody());

  // Convert parameter types
  llvm::SmallVector<mlir::Type, 4> argTypes;
  for (auto *param : decl->parameters()) {
    mlir::Type paramType = convertType(builder, param->getType());
    argTypes.push_back(paramType);
  }

  // Convert return type
  mlir::Type returnType = convertType(builder, decl->getReturnType());

  llvm::SmallVector<mlir::Type, 1> returnTypes;
  if (!mlir::isa<mlir::NoneType>(returnType)) {
    returnTypes.push_back(returnType);
  }

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
  }

  // Traverse function body manually
  TraverseStmt(decl->getBody());

  // Return true, but we've already handled traversal manually
  return true;
}

bool CMLIRCASTVisitor::TraverseStmt(clang::Stmt *stmt) {
  if (!stmt || !currentFunc) {
    return RecursiveASTVisitor::TraverseStmt(stmt);
  }

  if (auto *expr = llvm::dyn_cast<clang::Expr>(stmt)) {
    if (hasSideEffects(expr)) {
      generateExpr(expr);
      return true;
    }
  }

  return RecursiveASTVisitor::TraverseStmt(stmt);
}

bool CMLIRCASTVisitor::TraverseVarDecl(VarDecl *decl) {
  if (decl->isImplicit()) {
    return true;
  }

  if (llvm::isa<clang::ParmVarDecl>(decl)) {
    return true;
  }

  if (!currentFunc) {
    return true;
  }

  SourceManager &SM = context_manager_.ClangContext().getSourceManager();
  SourceLocation loc = decl->getLocation();

  auto mlirLoc = mlir::FileLineColLoc::get(
      &context_manager_.MLIRContext(), SM.getFilename(loc),
      SM.getSpellingLineNumber(loc), SM.getSpellingColumnNumber(loc));

  mlir::OpBuilder &builder = context_manager_.Builder();

  QualType clangType = decl->getType();
  mlir::Type mlirType = convertType(builder, clangType);

  mlir::Type allocaType;
  if (auto memrefType = mlir::dyn_cast<mlir::MemRefType>(mlirType)) {
    allocaType = memrefType;
  } else {
    allocaType = mlir::MemRefType::get({}, mlirType);
  }

  // Create alloca
  auto allocaOp = mlir::memref::AllocaOp::create(
      builder, mlirLoc, mlir::dyn_cast<mlir::MemRefType>(allocaType));

  symbolTable[decl] = allocaOp.getResult();

  if (decl->hasInit()) {
    Expr *init = decl->getInit();
    mlir::Value initValue = generateExpr(init);

    if (initValue) {
      mlir::memref::StoreOp::create(builder, mlirLoc, initValue,
                                    allocaOp.getResult(), mlir::ValueRange{});
    }
  }

  return true;
}

bool CMLIRCASTVisitor::TraverseReturnStmt(clang::ReturnStmt *stmt) {
  if (!currentFunc) {
    return true;
  }

  mlir::OpBuilder &builder = context_manager_.Builder();

  mlir::Value retValue = nullptr;
  if (auto *retExpr = stmt->getRetValue()) {
    retValue = generateExpr(retExpr);
  }

  if (retValue) {
    mlir::func::ReturnOp::create(builder, builder.getUnknownLoc(),
                                 mlir::ValueRange{retValue});
  } else {
    mlir::func::ReturnOp::create(builder, builder.getUnknownLoc());
  }

  return true;
}

bool CMLIRCASTVisitor::hasSideEffects(clang::Expr *expr) const {
  if (auto *binOp = llvm::dyn_cast<clang::BinaryOperator>(expr)) {
    return binOp->isAssignmentOp() || binOp->isCompoundAssignmentOp();
  }

  if (llvm::isa<clang::CallExpr>(expr)) {
    return true;
  }

  if (auto *unOp = llvm::dyn_cast<clang::UnaryOperator>(expr)) {
    return unOp->isIncrementDecrementOp();
  }

  return false;
}

} // namespace cmlirc
