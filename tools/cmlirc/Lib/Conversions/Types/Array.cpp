#include "../../Converter.h"

namespace cmlirc {

auto CMLIRConverter::convertArrayType(const clang::ArrayType *type)
    -> mlir::Type {
  llvm::SmallVector<int64_t, 4> dimensions;
  clang::QualType currentType = clang::QualType(type, 0);

  while (auto *arrayType =
             mlir::dyn_cast<clang::ArrayType>(currentType.getTypePtr())) {
    if (auto *constArrayType =
            mlir::dyn_cast<clang::ConstantArrayType>(arrayType)) {
      int64_t size = constArrayType->getSize().getSExtValue();
      dimensions.push_back(size);
      currentType = constArrayType->getElementType();
    } else {
      dimensions.push_back(-1);
      currentType = arrayType->getElementType();
    }
  }

  mlir::Type elementType = convertType(currentType);

  return mlir::MemRefType::get(dimensions, elementType);
}
} // namespace cmlirc
