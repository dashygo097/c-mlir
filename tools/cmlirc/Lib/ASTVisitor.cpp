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

  if (options::Verbose)
    llvm::outs() << "\n=== Processing Function: " << decl->getNameAsString()
                 << " ===\n";

  mlir::OpBuilder &builder = context_manager_.Builder();
  builder.setInsertionPointToEnd(context_manager_.Module().getBody());

  // Convert parameter types
  llvm::SmallVector<mlir::Type, 4> argTypes;
  for (auto *param : decl->parameters()) {
    mlir::Type paramType = convertType(builder, param->getType());
    argTypes.push_back(paramType);
    if (options::Verbose)
      llvm::outs() << "  Parameter: " << param->getNameAsString() << " : "
                   << param->getType().getAsString() << "\n";
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
    if (options::Verbose)
      llvm::outs() << "  Mapped param: " << param->getNameAsString()
                   << " -> %arg" << i << "\n";
  }

  // Traverse function body manually
  if (options::Verbose)
    llvm::outs() << "  Processing function body...\n";
  TraverseStmt(decl->getBody());

  if (options::Verbose)
    llvm::outs() << "=== Finished Function ===\n\n";

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

  if (options::Verbose)
    llvm::outs() << "  Local variable: " << decl->getNameAsString() << " : "
                 << decl->getType().getAsString();

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
    if (options::Verbose)
      llvm::outs() << " -> ";
    if (options::Verbose)
      allocaType.print(llvm::outs());
    if (options::Verbose)
      llvm::outs() << "\n";
  } else {
    allocaType = mlir::MemRefType::get({}, mlirType);
    if (options::Verbose)
      llvm::outs() << " -> memref<";
    if (options::Verbose)
      mlirType.print(llvm::outs());
    if (options::Verbose)
      llvm::outs() << ">\n";
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

  if (options::Verbose)
    llvm::outs() << "  Return statement\n";

  mlir::OpBuilder &builder = context_manager_.Builder();

  mlir::Value retValue = nullptr;
  if (auto *retExpr = stmt->getRetValue()) {
    if (options::Verbose)
      llvm::outs() << "    Generating return expression...\n";
    retValue = generateExpr(retExpr);
  }

  if (retValue) {
    mlir::func::ReturnOp::create(builder, builder.getUnknownLoc(),
                                 mlir::ValueRange{retValue});
    if (options::Verbose)
      llvm::outs() << "    Generated: func.return with value\n";
  } else {
    mlir::func::ReturnOp::create(builder, builder.getUnknownLoc());
    if (options::Verbose)
      llvm::outs() << "    Generated: func.return (void)\n";
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
