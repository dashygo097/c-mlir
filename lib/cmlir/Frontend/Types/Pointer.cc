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

  llvm::SmallVector<int64_t, 4> shape{mlir::ShapedType::kDynamic};
  mlir::Type elementType;
  if (pointeeType->isArrayType()) {
    auto lowered = convertType(pointeeType);
    if (!lowered) {
      return {};
    }
    auto array = mlir::dyn_cast<mlir::MemRefType>(lowered);
    if (!array) {
      return mlir::LLVM::LLVMPointerType::get(builder.getContext());
    }
    shape.append(array.getShape().begin(), array.getShape().end());
    elementType = array.getElementType();
  } else {
    elementType = convertType(pointeeType);
  }
  if (!elementType) {
    return {};
  }
  if (!mlir::MemRefType::isValidElementType(elementType)) {
    context.source().error({}, "unsupported pointer element representation");
    return {};
  }
  auto contiguous = mlir::MemRefType::get(shape, elementType);
  // Pointer arithmetic and rebinding preserve a descriptor's actual offset.
  auto layout = mlir::StridedLayoutAttr::get(
      builder.getContext(), mlir::ShapedType::kDynamic,
      contiguous.getStridesAndOffset().first);
  return mlir::MemRefType::get(shape, elementType, layout);
}

} // namespace cmlir::frontend
