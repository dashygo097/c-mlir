#include "../../Converter.h"
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
  mlir::Type mlirType = convertType(clangType);

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

  if (clangType->isStructureType()) {
    auto structType = mlir::dyn_cast<mlir::LLVM::LLVMStructType>(mlirType);
    if (!structType) {
      llvm::errs() << "Expected struct type\n";
      return false;
    }

    if (decl->hasInit()) {
      clang::Expr *init = decl->getInit();

      if (auto *initList = llvm::dyn_cast<clang::InitListExpr>(init)) {
        const clang::RecordType *recordType = clangType->getAsStructureType();
        if (!recordType) {
          llvm::errs() << "Expected record type\n";
          return false;
        }

        const clang::RecordDecl *recordDecl = recordType->getDecl();
        unsigned fieldIndex = 0;
        for (auto *_ : recordDecl->fields()) {
          if (fieldIndex >= initList->getNumInits())
            break;

          clang::Expr *initExpr = initList->getInit(fieldIndex);
          mlir::Value initValue = generateExpr(initExpr);

          if (!initValue) {
            llvm::errs() << "Failed to generate init value for field\n";
            fieldIndex++;
            continue;
          }

          mlir::Value loadedStruct = mlir::memref::LoadOp::create(
              builder, mlirLoc, allocaOp.getResult());

          mlir::Value updatedStruct = mlir::LLVM::InsertValueOp::create(
              builder, mlirLoc, loadedStruct, initValue, fieldIndex);

          mlir::memref::StoreOp::create(builder, mlirLoc, updatedStruct,
                                        allocaOp.getResult());

          fieldIndex++;
        }
      } else {
        mlir::Value initValue = generateExpr(init);
        if (initValue) {
          mlir::memref::StoreOp::create(builder, mlirLoc, initValue,
                                        allocaOp.getResult(),
                                        mlir::ValueRange{});
        }
      }
    }

    return true;
  }

  return RecursiveASTVisitor::TraverseVarDecl(decl);
}

} // namespace cmlirc
