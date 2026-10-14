#include "TypeLowering.h"

namespace cmlir::frontend {

auto TypeLowering::getFieldIndex(const clang::RecordDecl *recordDecl,
                                 const clang::FieldDecl *fieldDecl)
    -> std::optional<uint32_t> {
  if (!recordDecl || !fieldDecl) {
    return std::nullopt;
  }

  const clang::RecordDecl *definition = recordDecl->getDefinition();
  if (!definition) {
    definition = recordDecl;
  }

  auto it = recordFieldTable.find(definition);
  if (it == recordFieldTable.end()) {
    std::vector<const clang::FieldDecl *> fields;
    for (auto *field : definition->fields()) {
      fields.push_back(field);
    }

    recordFieldTable[definition] = std::move(fields);
    it = recordFieldTable.find(definition);
  }

  const clang::FieldDecl *canonicalField = fieldDecl->getCanonicalDecl();

  const auto &fields = it->second;
  for (uint32_t i = 0; i < fields.size(); ++i) {
    if (fields[i] == fieldDecl ||
        fields[i]->getCanonicalDecl() == canonicalField) {
      return i;
    }
  }

  return std::nullopt;
}

} // namespace cmlir::frontend
