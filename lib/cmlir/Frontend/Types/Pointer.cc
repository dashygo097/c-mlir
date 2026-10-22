#include "TypeLowering.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

namespace cmlir::frontend {

auto TypeLowering::convertPointerType(const clang::PointerType *type)
    -> mlir::Type {
  mlir::OpBuilder &builder = context.builder();
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
    auto lowered = convertType(pointeeType);
    if (!lowered) {
      return {};
    }
    auto array = mlir::dyn_cast<mlir::MemRefType>(lowered);
    if (!array) {
      return mlir::LLVM::LLVMPointerType::get(builder.getContext());
    }
    llvm::SmallVector<int64_t, 4> dimensions(array.getShape());
    dimensions.insert(dimensions.begin(), mlir::ShapedType::kDynamic);
    return mlir::MemRefType::get(dimensions, array.getElementType());
  }

  mlir::Type elementType = convertType(pointeeType);
  if (!elementType) {
    return nullptr;
  }

  if (!mlir::MemRefType::isValidElementType(elementType)) {
    context.source().error({}, "unsupported pointer element representation");
    return {};
  }
  return mlir::MemRefType::get({mlir::ShapedType::kDynamic}, elementType);
}

} // namespace cmlir::frontend
