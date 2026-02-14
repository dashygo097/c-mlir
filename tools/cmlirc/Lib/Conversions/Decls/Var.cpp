#include "../../Converter.h"
#include "../Types/Types.h"
#include "clang/Basic/SourceManager.h"

namespace cmlirc {

bool CMLIRConverter::TraverseVarDecl(clang::VarDecl *decl) {
  if (decl->isImplicit()) {
    return true;
  }

  if (llvm::isa<clang::ParmVarDecl>(decl)) {
    return true;
  }

  if (!currentFunc) {
    return true;
  }

  clang::SourceManager &SM = context_manager_.ClangContext().getSourceManager();
  clang::SourceLocation loc = decl->getLocation();

  auto mlirLoc = mlir::FileLineColLoc::get(
      &context_manager_.MLIRContext(), SM.getFilename(loc),
      SM.getSpellingLineNumber(loc), SM.getSpellingColumnNumber(loc));

  mlir::OpBuilder &builder = context_manager_.Builder();

  clang::QualType clangType = decl->getType();
  mlir::Type mlirType = convertType(builder, clangType);

  mlir::Type allocaType;
  if (auto memrefType = mlir::dyn_cast<mlir::MemRefType>(mlirType)) {
    allocaType = memrefType;
  } else {
    allocaType = mlir::MemRefType::get({}, mlirType);
  }

  auto allocaOp = mlir::memref::AllocaOp::create(
      builder, mlirLoc, mlir::dyn_cast<mlir::MemRefType>(allocaType));

  symbolTable[decl] = allocaOp.getResult();

  if (decl->hasInit()) {
    clang::Expr *init = decl->getInit();

    if (auto *initList = llvm::dyn_cast<clang::InitListExpr>(init)) {
      storeInitListValues(initList, allocaOp.getResult());
    } else {
      mlir::Value initValue = generateExpr(init);
      if (initValue) {
        mlir::memref::StoreOp::create(builder, mlirLoc, initValue,
                                      allocaOp.getResult(), mlir::ValueRange{});
      }
    }
  }

  return true;
}

} // namespace cmlirc
