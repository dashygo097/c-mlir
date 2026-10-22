#ifndef CMLIR_FRONTEND_TYPE_LOWERING_H
#define CMLIR_FRONTEND_TYPE_LOWERING_H

#include "clang-mlir/Context.h"
#include "clang/AST/Decl.h"
#include "llvm/ADT/DenseMap.h"
#include <optional>
#include <vector>

namespace cmlir::frontend {

class TypeLowering {
public:
  explicit TypeLowering(clang_mlir::Context &context) : context(context) {}

  // type traits
  [[nodiscard]] auto convertType(clang::QualType type) -> mlir::Type;
  [[nodiscard]] auto convertBuiltinType(const clang::BuiltinType *type)
      -> mlir::Type;
  [[nodiscard]] auto convertArrayType(const clang::ArrayType *type)
      -> mlir::Type;
  [[nodiscard]] auto convertPointerType(const clang::PointerType *type)
      -> mlir::Type;
  [[nodiscard]] auto convertTypedefType(const clang::TypedefType *type)
      -> mlir::Type;
  [[nodiscard]] auto convertRecordType(const clang::RecordType *type)
      -> mlir::Type;
  [[nodiscard]] auto convertReferenceType(const clang::ReferenceType *type)
      -> mlir::Type;
  [[nodiscard]] auto convertEnumType(const clang::EnumType *type) -> mlir::Type;

  auto convertRecordFieldType(clang::QualType type) -> mlir::Type;
  auto convertStorageType(clang::QualType type) -> mlir::Type;
  auto getFieldIndex(const clang::RecordDecl *, const clang::FieldDecl *)
      -> std::optional<uint32_t>;

private:
  clang_mlir::Context &context;
  llvm::DenseMap<const clang::RecordDecl *, mlir::Type> recordTypeTable;
  llvm::DenseMap<const clang::RecordDecl *,
                 std::vector<const clang::FieldDecl *>>
      recordFieldTable;
};

} // namespace cmlir::frontend

#endif // CMLIR_FRONTEND_TYPE_LOWERING_H
