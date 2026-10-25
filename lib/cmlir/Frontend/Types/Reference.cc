#include "TypeLowering.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"

namespace cmlir::frontend {

auto TypeLowering::convertReferenceType(const clang::ReferenceType *type)
    -> mlir::Type {
  auto referred = type->getPointeeType();
  if (referred.isVolatileQualified() || referred->isAtomicType() ||
      referred.getAddressSpace() != clang::LangAS::Default) {
    context.source().error({}, "qualified reference bindings are unsupported");
    return {};
  }
  auto object = referred->isArrayType() ? convertObjectType(referred)
                                        : convertType(referred);
  if (!object) {
    return {};
  }
  auto memref = referred->isArrayType()
                    ? mlir::dyn_cast<mlir::MemRefType>(object)
                    : mlir::MemRefType{};
  if (!memref && mlir::MemRefType::isValidElementType(object)) {
    memref = mlir::MemRefType::get({}, object);
  }
  if (memref) {
    auto layout = mlir::StridedLayoutAttr::get(
        object.getContext(), mlir::ShapedType::kDynamic,
        memref.getStridesAndOffset().first);
    return mlir::MemRefType::get(memref.getShape(), memref.getElementType(),
                                 layout, memref.getMemorySpace());
  }
  return mlir::LLVM::LLVMPointerType::get(object.getContext());
}

} // namespace cmlir::frontend
