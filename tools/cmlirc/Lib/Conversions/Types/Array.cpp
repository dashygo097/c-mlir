#include "../../Converter.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "clang/AST/Type.h"
#include "llvm/Support/WithColor.h"

namespace cmlirc {

static auto arrayLeafIsRecord(clang::QualType type) -> bool {
  type = type.getCanonicalType();

  while (auto *arrayType =
             mlir::dyn_cast<clang::ArrayType>(type.getTypePtr())) {
    type = arrayType->getElementType().getCanonicalType();
  }

  return mlir::isa<clang::RecordType>(type.getTypePtr());
}

auto CMLIRConverter::convertArrayType(const clang::ArrayType *type)
    -> mlir::Type {
  if (!type) {
    return nullptr;
  }

  clang::QualType arrayQualType = clang::QualType(type, 0).getCanonicalType();

  if (arrayLeafIsRecord(arrayQualType)) {
    llvm::SmallVector<uint64_t, 4> dimensions;
    clang::QualType currentType = arrayQualType;

    while (auto *arrayType =
               mlir::dyn_cast<clang::ArrayType>(currentType.getTypePtr())) {
      auto *constArrayType =
          mlir::dyn_cast<clang::ConstantArrayType>(arrayType);

      if (!constArrayType) {
        llvm::WithColor::error()
            << "cmlirc: variable-length array of record is not supported\n";
        return nullptr;
      }

      dimensions.push_back(constArrayType->getSize().getZExtValue());
      currentType = constArrayType->getElementType().getCanonicalType();
    }

    mlir::Type elementType = convertType(currentType);
    if (!elementType) {
      return nullptr;
    }

    mlir::Type resultType = elementType;
    for (auto it = dimensions.rbegin(); it != dimensions.rend(); ++it) {
      resultType = mlir::LLVM::LLVMArrayType::get(resultType, *it);
    }

    return resultType;
  }

  llvm::SmallVector<int64_t, 4> dimensions;
  clang::QualType currentType = arrayQualType;

  while (auto *arrayType =
             mlir::dyn_cast<clang::ArrayType>(currentType.getTypePtr())) {
    if (auto *constArrayType =
            mlir::dyn_cast<clang::ConstantArrayType>(arrayType)) {
      dimensions.push_back(constArrayType->getSize().getSExtValue());
      currentType = constArrayType->getElementType().getCanonicalType();
      continue;
    }

    dimensions.push_back(mlir::ShapedType::kDynamic);
    currentType = arrayType->getElementType().getCanonicalType();
  }

  mlir::Type elementType = convertType(currentType);
  if (!elementType) {
    return nullptr;
  }

  return mlir::MemRefType::get(dimensions, elementType);
}

} // namespace cmlirc
