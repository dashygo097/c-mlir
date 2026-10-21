#include "../Lowering.h"
#include "clang-mlir/Casts.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "clang/AST/ExprCXX.h"
#include "clang/Basic/SourceManager.h"
#include "llvm/Support/WithColor.h"

namespace cmlir::frontend {

auto isNoOpAggregateArrayInitializer(clang::Expr *init) -> bool {
  if (!init) {
    return true;
  }

  init = init->IgnoreParenImpCasts();

  if (auto *cleanups = mlir::dyn_cast<clang::ExprWithCleanups>(init)) {
    return isNoOpAggregateArrayInitializer(cleanups->getSubExpr());
  }

  if (auto *bindTemp = mlir::dyn_cast<clang::CXXBindTemporaryExpr>(init)) {
    return isNoOpAggregateArrayInitializer(bindTemp->getSubExpr());
  }

  if (auto *materialize =
          mlir::dyn_cast<clang::MaterializeTemporaryExpr>(init)) {
    return isNoOpAggregateArrayInitializer(materialize->getSubExpr());
  }

  if (auto *defaultInit = mlir::dyn_cast<clang::CXXDefaultInitExpr>(init)) {
    return isNoOpAggregateArrayInitializer(defaultInit->getExpr());
  }

  if (auto *constructExpr = mlir::dyn_cast<clang::CXXConstructExpr>(init)) {
    return constructExpr->getNumArgs() == 0;
  }

  if (mlir::isa<clang::ImplicitValueInitExpr>(init)) {
    return true;
  }

  if (auto *initList = mlir::dyn_cast<clang::InitListExpr>(init)) {
    return initList->getNumInits() == 0;
  }

  return false;
}

auto ASTLowering::emitVariable(clang::VarDecl *decl) -> bool {
  if (decl->isImplicit()) {
    return true;
  }
  if (mlir::isa<clang::ParmVarDecl>(decl)) {
    return true;
  }
  if (decl->isStaticLocal()) {
    context.source().error(decl->getLocation(),
                           "static local storage is unsupported");
    return false;
  }
  if (!function.operation) {
    return true;
  }

  clang::SourceManager &sm = context.ast().getSourceManager();
  clang::SourceLocation loc = decl->getLocation();
  auto mlirLoc = mlir::FileLineColLoc::get(&context.ir(), sm.getFilename(loc),
                                           sm.getSpellingLineNumber(loc),
                                           sm.getSpellingColumnNumber(loc));

  mlir::OpBuilder &builder = context.builder();
  clang::QualType clangType = decl->getType();

  if (clangType->isPointerType()) {
    auto ptrType = mlir::LLVM::LLVMPointerType::get(builder.getContext());
    auto one = clang_mlir::intConst(builder, mlirLoc, builder.getI64Type(), 1);

    auto allocaOp = mlir::LLVM::AllocaOp::create(builder, mlirLoc, ptrType,
                                                 ptrType, one, 0);
    function.locals[decl] = allocaOp.getResult();

    if (decl->hasInit()) {
      mlir::Value initValue = generateExpr(decl->getInit());

      if (!initValue) {
        return false;
      }

      if (!mlir::isa<mlir::LLVM::LLVMPointerType>(initValue.getType())) {
        initValue = clang_mlir::toPointer(builder, mlirLoc, initValue, ptrType);
      }

      if (!initValue) {
        llvm::WithColor::error()
            << "clang-to-mlir: failed to initialize pointer variable: "
            << decl->getNameAsString() << "\n";
        return false;
      }

      mlir::LLVM::StoreOp::create(builder, mlirLoc, initValue,
                                  allocaOp.getResult());
    }

    return true;
  }

  if (clangType->isStructureType()) {
    mlir::Type mlirType = types.convertType(clangType);
    auto structType = mlir::dyn_cast<mlir::LLVM::LLVMStructType>(mlirType);
    if (!structType) {
      llvm::WithColor::error() << "clang-to-mlir: expected struct type\n";
      return false;
    }

    auto ptrType = mlir::LLVM::LLVMPointerType::get(builder.getContext());
    auto one = clang_mlir::intConst(builder, mlirLoc, builder.getI64Type(), 1);
    auto allocaOp = mlir::LLVM::AllocaOp::create(builder, mlirLoc, ptrType,
                                                 structType, one, 0);
    function.locals[decl] = allocaOp.getResult();

    if (decl->hasInit()) {
      clang::Expr *init = decl->getInit();

      if (auto *initList = mlir::dyn_cast<clang::InitListExpr>(init)) {
        const clang::RecordDecl *recordDecl =
            clangType->getAsStructureType()->getDecl();

        uint32_t fieldIndex = 0;
        for (auto *_ : recordDecl->fields()) {
          if (fieldIndex >= initList->getNumInits()) {
            break;
          }

          mlir::Value initValue = generateExpr(initList->getInit(fieldIndex));
          if (!initValue) {
            return false;
          }

          auto zero =
              clang_mlir::intConst(builder, mlirLoc, builder.getI32Type(), 0);
          auto fieldIdx = mlir::LLVM::ConstantOp::create(
              builder, mlirLoc, builder.getI32Type(),
              builder.getI32IntegerAttr(fieldIndex));

          auto fieldPtr = mlir::LLVM::GEPOp::create(
              builder, mlirLoc, ptrType, structType, allocaOp.getResult(),
              llvm::SmallVector<mlir::Value, 2>{zero, fieldIdx});

          mlir::LLVM::StoreOp::create(builder, mlirLoc, initValue, fieldPtr);
          fieldIndex++;
        }
      } else {
        mlir::Value initValue = generateExpr(init);
        if (!initValue) {
          return false;
        }

        mlir::LLVM::StoreOp::create(builder, mlirLoc, initValue,
                                    allocaOp.getResult());
      }
    }

    return true;
  }

  {
    clang::QualType canonical = clangType.getCanonicalType();
    const clang::VariableArrayType *vat = nullptr;

    clang::QualType cur = canonical;
    while (auto *arrayType =
               mlir::dyn_cast_or_null<clang::ArrayType>(cur.getTypePtr())) {
      if (auto *vla = mlir::dyn_cast<clang::VariableArrayType>(arrayType)) {
        vat = vla;
        break;
      }

      if (auto *constantArray =
              mlir::dyn_cast<clang::ConstantArrayType>(arrayType)) {
        cur = constantArray->getElementType().getCanonicalType();
      } else {
        break;
      }
    }

    if (vat) {
      llvm::SmallVector<mlir::Value> dynamicSizes;
      llvm::SmallVector<int64_t> shape;

      cur = canonical;
      while (auto *arrayType =
                 mlir::dyn_cast_or_null<clang::ArrayType>(cur.getTypePtr())) {
        if (auto *vla = mlir::dyn_cast<clang::VariableArrayType>(arrayType)) {
          mlir::Value dimSize = generateExpr(vla->getSizeExpr());
          dimSize = clang_mlir::toIndex(builder, mlirLoc, dimSize);

          if (!dimSize) {
            return false;
          }

          dynamicSizes.push_back(dimSize);
          shape.push_back(mlir::ShapedType::kDynamic);
          cur = vla->getElementType().getCanonicalType();
        } else if (auto *constantArray =
                       mlir::dyn_cast<clang::ConstantArrayType>(arrayType)) {
          shape.push_back(constantArray->getSize().getSExtValue());
          cur = constantArray->getElementType().getCanonicalType();
        } else {
          break;
        }
      }

      mlir::Type elemType = types.convertType(cur);
      if (!elemType) {
        llvm::WithColor::error() << "clang-to-mlir: failed to convert VLA "
                                    "element type for variable: "
                                 << decl->getNameAsString() << "\n";
        return false;
      }

      auto memrefType = mlir::MemRefType::get(shape, elemType);
      auto allocaOp = mlir::memref::AllocaOp::create(builder, mlirLoc,
                                                     memrefType, dynamicSizes);
      function.locals[decl] = allocaOp.getResult();

      if (decl->hasInit()) {
        mlir::Value initValue = generateExpr(decl->getInit());
        if (!initValue) {
          return false;
        }

        mlir::memref::StoreOp::create(builder, mlirLoc, initValue,
                                      allocaOp.getResult(), mlir::ValueRange{});
      }

      return true;
    }
  }

  mlir::Type mlirType = types.convertType(clangType);
  if (!mlirType) {
    llvm::WithColor::error()
        << "clang-to-mlir: failed to convert type for variable: "
        << decl->getNameAsString() << "\n";
    return false;
  }

  if (auto llvmArrayType =
          mlir::dyn_cast<mlir::LLVM::LLVMArrayType>(mlirType)) {
    auto ptrType = mlir::LLVM::LLVMPointerType::get(builder.getContext());
    auto one = clang_mlir::intConst(builder, mlirLoc, builder.getI64Type(), 1);

    auto allocaOp = mlir::LLVM::AllocaOp::create(builder, mlirLoc, ptrType,
                                                 llvmArrayType, one, 0);
    function.locals[decl] = allocaOp.getResult();

    if (decl->hasInit() && !isNoOpAggregateArrayInitializer(decl->getInit())) {
      llvm::WithColor::error() << "clang-to-mlir: explicit initializer for "
                                  "LLVM aggregate array is not "
                                  "supported yet: "
                               << decl->getNameAsString() << "\n";
      return false;
    }

    return true;
  }

  mlir::MemRefType allocaType;

  if (auto memrefType = mlir::dyn_cast<mlir::MemRefType>(mlirType)) {
    allocaType = memrefType;
  } else {
    allocaType = mlir::MemRefType::get({}, mlirType);
  }

  auto allocaOp = mlir::memref::AllocaOp::create(builder, mlirLoc, allocaType);
  function.locals[decl] = allocaOp.getResult();

  if (decl->hasInit()) {
    clang::Expr *init = decl->getInit();

    if (auto *initList = mlir::dyn_cast<clang::InitListExpr>(init)) {
      if (!storeInitListValues(initList, allocaOp.getResult())) {
        return false;
      }
    } else {
      mlir::Value initValue = generateExpr(init);
      if (!initValue) {
        return false;
      }

      if (initValue.getType() != mlirType) {
        if (mlir::Value casted =
                clang_mlir::toValue(builder, mlirLoc, initValue, mlirType)) {
          initValue = casted;
        }
      }

      mlir::memref::StoreOp::create(builder, mlirLoc, initValue,
                                    allocaOp.getResult(), mlir::ValueRange{});
    }
  }

  return true;
}

} // namespace cmlir::frontend
