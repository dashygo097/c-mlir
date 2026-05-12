#include "../../Converter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

namespace cmlirc {

auto CMLIRConverter::convertPointerType(const clang::PointerType *type)
    -> mlir::Type {
  mlir::OpBuilder &builder = contextManager.Builder();
  clang::QualType pointeeType = type->getPointeeType().getCanonicalType();

  if (pointeeType->isVoidType()) {
    return mlir::LLVM::LLVMPointerType::get(builder.getContext());
  }

  if (pointeeType->isFunctionType()) {
    return mlir::LLVM::LLVMPointerType::get(builder.getContext());
  }

  if (mlir::isa<clang::RecordType>(pointeeType.getTypePtr())) {
    return mlir::LLVM::LLVMPointerType::get(builder.getContext());
  }

  if (mlir::isa<clang::ArrayType>(pointeeType.getTypePtr())) {
    llvm::SmallVector<int64_t, 4> dimensions;
    clang::QualType currentType = pointeeType;

    while (auto *arrType =
               mlir::dyn_cast<clang::ArrayType>(currentType.getTypePtr())) {
      if (auto *constArrayType =
              mlir::dyn_cast<clang::ConstantArrayType>(arrType)) {
        dimensions.push_back(constArrayType->getSize().getSExtValue());
        currentType = constArrayType->getElementType().getCanonicalType();
      } else {
        dimensions.push_back(mlir::ShapedType::kDynamic);
        currentType = arrType->getElementType().getCanonicalType();
      }
    }

    mlir::Type elementType = convertType(currentType);
    if (!elementType) {
      return nullptr;
    }

    dimensions.insert(dimensions.begin(), mlir::ShapedType::kDynamic);
    return mlir::MemRefType::get(dimensions, elementType);
  }

  mlir::Type elementType = convertType(pointeeType);
  if (!elementType) {
    return nullptr;
  }

  return mlir::MemRefType::get({mlir::ShapedType::kDynamic}, elementType);
}

} // namespace cmlirc
