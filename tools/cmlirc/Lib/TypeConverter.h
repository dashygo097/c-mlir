#ifndef CMLIRC_TYPE_CONVERTER_H
#define CMLIRC_TYPE_CONVERTER_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/Types.h"
#include "clang/AST/Type.h"

namespace cmlirc {

class TypeConverter {
public:
  explicit TypeConverter(mlir::OpBuilder &builder);
  ~TypeConverter() = default;

  [[nodiscard]] mlir::Type convertType(clang::QualType type);

private:
  mlir::OpBuilder &builder_;

  [[nodiscard]] mlir::Type convertBuiltinType(const clang::BuiltinType *type);
  [[nodiscard]] mlir::Type convertPointerType(const clang::PointerType *type);
  [[nodiscard]] mlir::Type convertArrayType(const clang::ArrayType *type);
};

} // namespace cmlirc

#endif // CMLIRC_TYPE_CONVERTER_H
