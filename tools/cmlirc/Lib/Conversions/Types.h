#ifndef CMLIRC_TYPE_CONVERTER_H
#define CMLIRC_TYPE_CONVERTER_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/Types.h"
#include "clang/AST/Type.h"

namespace cmlirc {

[[nodiscard]] mlir::Type convertType(mlir::OpBuilder &builder,
                                     clang::QualType type);
[[nodiscard]] mlir::Type convertBuiltinType(mlir::OpBuilder &builder,
                                            const clang::BuiltinType *type);
[[nodiscard]] mlir::Type convertArrayType(mlir::OpBuilder &builder,
                                          const clang::ArrayType *type);
[[nodiscard]] mlir::Type convertPointerType(mlir::OpBuilder &builder,
                                            const clang::PointerType *type);

} // namespace cmlirc

#endif // CMLIRC_TYPE_CONVERTER_H
