#include "../../Converter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "clang/Basic/SourceManager.h"

namespace cmlirc {

bool CMLIRConverter::TraverseVarDecl(clang::VarDecl *decl) {
  if (decl->isImplicit()) {
    return true;
  }

  if (mlir::isa<clang::ParmVarDecl>(decl)) {
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

  if (!mlirType) {
    llvm::errs() << "Failed to convert type for variable: "
                 << decl->getNameAsString() << "\n";
    return false;
  }

  if (clangType->isPointerType()) {
    if (decl->hasInit()) {
      mlir::Value initValue = generateExpr(decl->getInit());
      lastArrayAccess_.reset();
      if (initValue)
        symbolTable[decl] = initValue;
    }
    return true;
  }

  if (clangType->isStructureType()) {
    auto structType = mlir::dyn_cast<mlir::LLVM::LLVMStructType>(mlirType);
    if (!structType) {
      llvm::errs() << "Expected struct type\n";
      return false;
    }

    auto ptrType = mlir::LLVM::LLVMPointerType::get(builder.getContext());
    auto one = mlir::LLVM::ConstantOp::create(
        builder, mlirLoc, builder.getI64Type(), builder.getI64IntegerAttr(1));

    auto allocaOp = mlir::LLVM::AllocaOp::create(builder, mlirLoc, ptrType,
                                                 structType, one, 0);

    symbolTable[decl] = allocaOp.getResult();

    if (decl->hasInit()) {
      clang::Expr *init = decl->getInit();

      if (auto *initList = mlir::dyn_cast<clang::InitListExpr>(init)) {
        const clang::RecordType *recordType = clangType->getAsStructureType();
        if (!recordType) {
          llvm::errs() << "Expected record type\n";
          return false;
        }

        const clang::RecordDecl *recordDecl = recordType->getDecl();
        uint32_t fieldIndex = 0;

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

          auto zero = mlir::LLVM::ConstantOp::create(
              builder, mlirLoc, builder.getI32Type(),
              builder.getI32IntegerAttr(0));
          auto fieldIdx = mlir::LLVM::ConstantOp::create(
              builder, mlirLoc, builder.getI32Type(),
              builder.getI32IntegerAttr(fieldIndex));

          llvm::SmallVector<mlir::Value, 2> indices;
          indices.push_back(zero);
          indices.push_back(fieldIdx);

          auto fieldPtr =
              mlir::LLVM::GEPOp::create(builder, mlirLoc, ptrType, structType,
                                        allocaOp.getResult(), indices);

          mlir::LLVM::StoreOp::create(builder, mlirLoc, initValue, fieldPtr);

          fieldIndex++;
        }
      } else {
        mlir::Value initValue = generateExpr(init);
        if (initValue) {
          mlir::LLVM::StoreOp::create(builder, mlirLoc, initValue, allocaOp);
        }
      }
    }

    return true;
  }

  mlir::Type allocaType;
  if (auto memrefType = mlir::dyn_cast<mlir::MemRefType>(mlirType)) {
    allocaType = memrefType;
  } else {
    allocaType = mlir::MemRefType::get({}, mlirType);
  }

  auto allocaOp = mlir::memref::AllocaOp::create(
      builder, mlirLoc, mlir::cast<mlir::MemRefType>(allocaType));

  symbolTable[decl] = allocaOp.getResult();

  if (decl->hasInit()) {
    clang::Expr *init = decl->getInit();

    if (auto *initList = mlir::dyn_cast<clang::InitListExpr>(init)) {
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
