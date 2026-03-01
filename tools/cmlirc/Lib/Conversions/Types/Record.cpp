#include "../../Converter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

namespace cmlirc {

mlir::Type CMLIRConverter::convertRecordType(const clang::RecordType *type) {
  mlir::OpBuilder &builder = context_manager_.Builder();
  const clang::RecordDecl *recordDecl = type->getDecl();

  auto it = recordTypeTable.find(recordDecl);
  if (it != recordTypeTable.end()) {
    return it->second;
  }

  if (!recordDecl->isCompleteDefinition()) {
    llvm::errs() << "Incomplete struct definition: "
                 << recordDecl->getNameAsString() << "\n";
    return nullptr;
  }

  llvm::SmallVector<mlir::Type, 8> memberTypes;

  for (auto *field : recordDecl->fields()) {
    mlir::Type fieldType = convertType(field->getType());
    if (!fieldType) {
      llvm::errs() << "Failed to convert field type: "
                   << field->getNameAsString() << "\n";
      return nullptr;
    }
    memberTypes.push_back(fieldType);
  }

  auto structType =
      mlir::LLVM::LLVMStructType::getLiteral(builder.getContext(), memberTypes);

  recordTypeTable[recordDecl] = structType;

  return structType;
}
} // namespace cmlirc
