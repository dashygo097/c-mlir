#include "../../Converter.h"
#include "../Utils/Casts.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "clang/Basic/SourceManager.h"

namespace cmlirc {

bool CMLIRConverter::TraverseVarDecl(clang::VarDecl *decl) {
  if (decl->isImplicit())
    return true;
  if (mlir::isa<clang::ParmVarDecl>(decl))
    return true;
  if (!currentFunc)
    return true;

  clang::SourceManager &SM = context_manager_.ClangContext().getSourceManager();
  clang::SourceLocation loc = decl->getLocation();
  auto mlirLoc = mlir::FileLineColLoc::get(
      &context_manager_.MLIRContext(), SM.getFilename(loc),
      SM.getSpellingLineNumber(loc), SM.getSpellingColumnNumber(loc));

  mlir::OpBuilder &builder = context_manager_.Builder();
  clang::QualType clangType = decl->getType();

  if (clangType->isPointerType()) {
    if (decl->hasInit()) {
      mlir::Value initValue = generateExpr(decl->getInit());
      lastArrayAccess.reset();
      if (initValue)
        symbolTable[decl] = initValue;
    }
    return true;
  }

  if (clangType->isStructureType()) {
    mlir::Type mlirType = convertType(clangType);
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
        const clang::RecordDecl *recordDecl =
            clangType->getAsStructureType()->getDecl();
        uint32_t fieldIndex = 0;
        for (auto *_ : recordDecl->fields()) {
          if (fieldIndex >= initList->getNumInits())
            break;
          mlir::Value initValue = generateExpr(initList->getInit(fieldIndex));
          if (initValue) {
            auto zero = mlir::LLVM::ConstantOp::create(
                builder, mlirLoc, builder.getI32Type(),
                builder.getI32IntegerAttr(0));
            auto fieldIdx = mlir::LLVM::ConstantOp::create(
                builder, mlirLoc, builder.getI32Type(),
                builder.getI32IntegerAttr(fieldIndex));
            auto fieldPtr = mlir::LLVM::GEPOp::create(
                builder, mlirLoc, ptrType, structType, allocaOp.getResult(),
                llvm::SmallVector<mlir::Value, 2>{zero, fieldIdx});
            mlir::LLVM::StoreOp::create(builder, mlirLoc, initValue, fieldPtr);
          }
          fieldIndex++;
        }
      } else {
        mlir::Value initValue = generateExpr(init);
        if (initValue)
          mlir::LLVM::StoreOp::create(builder, mlirLoc, initValue, allocaOp);
      }
    }
    return true;
  }

  {
    clang::QualType canonical = clangType.getCanonicalType();
    const clang::VariableArrayType *vat = nullptr;

    clang::QualType cur = canonical;
    while (auto *av =
               mlir::dyn_cast_or_null<clang::ArrayType>(cur.getTypePtr())) {
      if (auto *v = mlir::dyn_cast<clang::VariableArrayType>(av)) {
        vat = v;
        break;
      }
      if (auto *c = mlir::dyn_cast<clang::ConstantArrayType>(av)) {
        cur = c->getElementType().getCanonicalType();
      } else {
        break;
      }
    }

    if (vat) {
      llvm::SmallVector<mlir::Value> dynamicSizes;
      llvm::SmallVector<int64_t> shape;
      cur = canonical;

      while (auto *av =
                 mlir::dyn_cast_or_null<clang::ArrayType>(cur.getTypePtr())) {
        if (auto *v = mlir::dyn_cast<clang::VariableArrayType>(av)) {
          mlir::Value dimSize = generateExpr(v->getSizeExpr());
          dimSize = detail::toIndex(builder, mlirLoc, dimSize);
          dynamicSizes.push_back(dimSize);
          shape.push_back(mlir::ShapedType::kDynamic);
          cur = v->getElementType().getCanonicalType();
        } else if (auto *c = mlir::dyn_cast<clang::ConstantArrayType>(av)) {
          shape.push_back(c->getSize().getSExtValue());
          cur = c->getElementType().getCanonicalType();
        } else {
          break;
        }
      }

      mlir::Type elemType = convertType(cur);
      auto memrefType = mlir::MemRefType::get(shape, elemType);
      auto allocaOp = mlir::memref::AllocaOp::create(builder, mlirLoc,
                                                     memrefType, dynamicSizes);
      symbolTable[decl] = allocaOp.getResult();

      if (decl->hasInit()) {
        mlir::Value initValue = generateExpr(decl->getInit());
        if (initValue)
          mlir::memref::StoreOp::create(builder, mlirLoc, initValue,
                                        allocaOp.getResult(),
                                        mlir::ValueRange{});
      }
      return true;
    }
  }

  mlir::Type mlirType = convertType(clangType);
  if (!mlirType) {
    llvm::errs() << "Failed to convert type for variable: "
                 << decl->getNameAsString() << "\n";
    return false;
  }

  mlir::Type allocaType;
  if (auto memrefType = mlir::dyn_cast<mlir::MemRefType>(mlirType))
    allocaType = memrefType;
  else
    allocaType = mlir::MemRefType::get({}, mlirType);

  auto allocaOp = mlir::memref::AllocaOp::create(
      builder, mlirLoc, mlir::cast<mlir::MemRefType>(allocaType));
  symbolTable[decl] = allocaOp.getResult();

  if (decl->hasInit()) {
    clang::Expr *init = decl->getInit();
    if (auto *initList = mlir::dyn_cast<clang::InitListExpr>(init)) {
      storeInitListValues(initList, allocaOp.getResult());
    } else {
      mlir::Value initValue = generateExpr(init);
      if (initValue)
        mlir::memref::StoreOp::create(builder, mlirLoc, initValue,
                                      allocaOp.getResult(), mlir::ValueRange{});
    }
  }

  return true;
}

} // namespace cmlirc
