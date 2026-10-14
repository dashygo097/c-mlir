#ifndef CLANG_MLIR_TYPES_H
#define CLANG_MLIR_TYPES_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"

namespace clang_mlir {

inline auto isIndexType(mlir::Type type) -> bool { return type.isIndex(); }

inline auto isIntegerType(mlir::Type type) -> bool {
  return mlir::isa<mlir::IntegerType>(type);
}

inline auto isIntegerLikeType(mlir::Type type) -> bool {
  return type.isIndex() || mlir::isa<mlir::IntegerType>(type);
}

inline auto isFloatType(mlir::Type type) -> bool {
  return mlir::isa<mlir::FloatType>(type);
}

inline auto isNumericType(mlir::Type type) -> bool {
  return isIntegerLikeType(type) || isFloatType(type);
}

} // namespace clang_mlir

#endif // CLANG_MLIR_TYPES_H
