#include "../../Converter.h"

namespace cmlirc {

auto CMLIRConverter::TraverseRecordDecl(clang::RecordDecl *decl) -> bool {
  if (!decl) {
    return true;
  }

  if (!decl->isCompleteDefinition()) {
    return true;
  }

  const clang::RecordDecl *definition = decl->getDefinition();
  if (!definition) {
    definition = decl;
  }

  std::vector<const clang::FieldDecl *> fields;
  for (auto *field : definition->fields()) {
    fields.push_back(field);
  }

  recordFieldTable[definition] = std::move(fields);

  return RecursiveASTVisitor::TraverseRecordDecl(decl);
}

} // namespace cmlirc
