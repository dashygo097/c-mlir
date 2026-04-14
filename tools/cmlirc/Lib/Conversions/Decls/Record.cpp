#include "../../Converter.h"

namespace cmlirc {

auto CMLIRConverter::TraverseRecordDecl(clang::RecordDecl *decl) -> bool {
  if (!decl->isCompleteDefinition()) {
    return true;
  }

  std::vector<const clang::FieldDecl *> fields;
  for (auto *field : decl->fields()) {
    fields.push_back(field);
  }
  recordFieldTable[decl] = fields;

  return RecursiveASTVisitor::TraverseRecordDecl(decl);
}

} // namespace cmlirc
