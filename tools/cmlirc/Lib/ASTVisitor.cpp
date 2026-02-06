#include "./ASTVisitor.h"
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
    return true;
  }

  if (!currentFunc) {
    return true;
  }

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
    llvm::outs() << " -> ";
    allocaType.print(llvm::outs());
    llvm::outs() << "\n";
  } else {
    allocaType = mlir::MemRefType::get({}, mlirType);
    llvm::outs() << " -> memref<";
    mlirType.print(llvm::outs());
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

} // namespace cmlirc
