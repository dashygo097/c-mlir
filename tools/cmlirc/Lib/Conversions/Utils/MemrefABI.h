#ifndef CMLIRC_MEMREF_ABI_H
#define CMLIRC_MEMREF_ABI_H

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/WithColor.h"

namespace cmlirc::utils {

inline auto memrefSameElementType(mlir::MemRefType src, mlir::MemRefType dst)
    -> bool {
  return src.getElementType() == dst.getElementType();
}

inline auto memrefIsScalarAddressToPointer(mlir::MemRefType src,
                                           mlir::MemRefType dst) -> bool {
  return src.getRank() == 0 && dst.getRank() == 1 &&
         memrefSameElementType(src, dst);
}

inline auto reinterpretScalarMemRefAsPointerMemRef(mlir::OpBuilder &builder,
                                                   mlir::Location loc,
                                                   mlir::Value value,
                                                   mlir::MemRefType dstType)
    -> mlir::Value {
  auto srcType = mlir::dyn_cast<mlir::MemRefType>(value.getType());
  if (!srcType || !memrefIsScalarAddressToPointer(srcType, dstType)) {
    return nullptr;
  }

  mlir::OpFoldResult offset = builder.getIndexAttr(0);

  llvm::SmallVector<mlir::OpFoldResult, 1> sizes;
  llvm::SmallVector<mlir::OpFoldResult, 1> strides;

  sizes.push_back(builder.getIndexAttr(1));
  strides.push_back(builder.getIndexAttr(1));

  return mlir::memref::ReinterpretCastOp::create(builder, loc, dstType, value,
                                                 offset, sizes, strides)
      .getResult();
}

inline auto coerceMemRefForCall(mlir::OpBuilder &builder, mlir::Location loc,
                                mlir::Value value, mlir::MemRefType dstType)
    -> mlir::Value {
  auto srcType = mlir::dyn_cast<mlir::MemRefType>(value.getType());
  if (!srcType) {
    return nullptr;
  }

  if (srcType == dstType) {
    return value;
  }

  if (!memrefSameElementType(srcType, dstType)) {
    return nullptr;
  }

  if (mlir::memref::CastOp::areCastCompatible(srcType, dstType)) {
    return mlir::memref::CastOp::create(builder, loc, dstType, value)
        .getResult();
  }

  if (mlir::Value reinterpreted = reinterpretScalarMemRefAsPointerMemRef(
          builder, loc, value, dstType)) {
    return reinterpreted;
  }

  return nullptr;
}

} // namespace cmlirc::utils

#endif // CMLIRC_MEMREF_ABI_H
