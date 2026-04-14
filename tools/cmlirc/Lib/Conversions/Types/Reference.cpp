#include "../../Converter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

namespace cmlirc {

auto CMLIRConverter::convertReferenceType(const clang::ReferenceType *type)
    -> mlir::Type {
  mlir::OpBuilder &builder = contextManager.Builder();

  clang::QualType pointeeType = type->getPointeeType().getCanonicalType();

  if (mlir::isa<clang::RecordType>(pointeeType.getTypePtr())) {
    return mlir::LLVM::LLVMPointerType::get(builder.getContext());
  }

  mlir::Type elementType = convertType(pointeeType);
  if (!elementType) {
    return nullptr;
  }
  return mlir::MemRefType::get({}, elementType);
}

} // namespace cmlirc
