#include "Initialization.h"
#include "Expressions/ExpressionLowering.h"
#include "clang-mlir/Casts.h"
#include "clang-mlir/Constants.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "clang/AST/DeclCXX.h"

namespace cmlir::frontend {

auto ObjectInitialization::emit(clang::QualType type, clang::Expr *initializer,
                                mlir::Value storage,
                                llvm::ArrayRef<mlir::Value> indices) -> bool {
  auto &builder = context.builder();
  auto sourceLoc =
      initializer ? initializer->getExprLoc() : clang::SourceLocation{};
  auto loc = context.source().location(sourceLoc);
  if (initializer) {
    initializer = initializer->IgnoreParens();
    if (auto *cleanup = llvm::dyn_cast<clang::ExprWithCleanups>(initializer)) {
      return emit(type, cleanup->getSubExpr(), storage, indices);
    }
    if (auto *defaultInit =
            llvm::dyn_cast<clang::CXXDefaultInitExpr>(initializer)) {
      return emit(type, defaultInit->getExpr(), storage, indices);
    }
    if (llvm::isa<clang::ImplicitValueInitExpr, clang::CXXScalarValueInitExpr>(
            initializer)) {
      initializer = nullptr;
    } else if (auto *construct =
                   llvm::dyn_cast<clang::CXXConstructExpr>(initializer)) {
      if (construct->getConstructor()->isTrivial() &&
          construct->getNumArgs() == 0) {
        if (!construct->requiresZeroInitialization()) {
          return true;
        }
        initializer = nullptr;
      }
    }
  }
  auto *list = llvm::dyn_cast_or_null<clang::InitListExpr>(initializer);
  if (list && list->getSemanticForm()) {
    list = list->getSemanticForm();
  }

  if (auto *array = context.ast().getAsConstantArrayType(type)) {
    if (initializer && !list) {
      context.source().error(
          sourceLoc, "array initialization requires an initializer list");
      return false;
    }
    auto elementType = array->getElementType();
    auto count = array->getSize().getZExtValue();
    for (uint64_t i = 0; i < count; ++i) {
      auto *element = list && i < list->getNumInits() ? list->getInit(i)
                      : list                          ? list->getArrayFiller()
                                                      : nullptr;
      auto elementStorage = storage;
      llvm::SmallVector<mlir::Value, 4> elementIndices(indices);
      if (llvm::isa<mlir::MemRefType>(storage.getType())) {
        elementIndices.push_back(clang_mlir::indexConst(builder, loc, i));
      } else {
        auto layout = types.convertStorageType(type);
        if (!layout) {
          return false;
        }
        elementStorage = mlir::LLVM::GEPOp::create(
            builder, loc, storage.getType(), layout, storage,
            llvm::ArrayRef<mlir::LLVM::GEPArg>{0, static_cast<int32_t>(i)});
      }
      if (!emit(elementType, element, elementStorage, elementIndices)) {
        return false;
      }
    }
    return true;
  }

  if (type->isRecordType() && (!initializer || list)) {
    auto layout = types.convertStorageType(type);
    if (!layout) {
      return false;
    }
    unsigned index = 0;
    for (auto *field : type->getAsRecordDecl()->getDefinition()->fields()) {
      auto *element =
          list && index < list->getNumInits() ? list->getInit(index) : nullptr;
      auto pointer = mlir::LLVM::GEPOp::create(
          builder, loc, storage.getType(), layout, storage,
          llvm::ArrayRef<mlir::LLVM::GEPArg>{0, static_cast<int32_t>(index)});
      if (!emit(field->getType(), element, pointer)) {
        return false;
      }
      ++index;
    }
    return true;
  }

  if (list) {
    return emit(type, list->getNumInits() ? list->getInit(0) : nullptr, storage,
                indices);
  }
  auto element = types.convertStorageType(type);
  if (auto memref = mlir::dyn_cast<mlir::MemRefType>(storage.getType())) {
    element = memref.getElementType();
  }
  if (!element) {
    return false;
  }
  mlir::Value value;
  if (initializer) {
    auto emitted = expressions.emitValue(*initializer);
    if (mlir::failed(emitted)) {
      return false;
    }
    value = *emitted;
    value = clang_mlir::toValue(
        builder, loc, value, element,
        initializer->getType()->isSignedIntegerOrEnumerationType());
  } else if (element.isIntOrFloat()) {
    value = clang_mlir::zeroConst(builder, loc, element);
  } else if (mlir::LLVM::isCompatibleType(element)) {
    value = mlir::LLVM::ZeroOp::create(builder, loc, element);
  }
  auto address = clang_mlir::Address::get(storage, element, indices);
  if (!value || mlir::failed(address)) {
    context.source().error(sourceLoc, "invalid object initializer");
    return false;
  }
  return mlir::succeeded(address->store(builder, loc, value));
}

} // namespace cmlir::frontend
