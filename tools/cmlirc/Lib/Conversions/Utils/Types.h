#ifndef CMLIRC_TYPES_H
#define CMLIRC_TYPES_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"

namespace cmlirc::utils {

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

} // namespace cmlirc::utils

#endif // CMLIRC_TYPES_H
