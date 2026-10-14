#include "clang-mlir/Address.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Diagnostics.h"

namespace clang_mlir {

auto Address::get(mlir::Value base, mlir::Type element,
                  llvm::ArrayRef<mlir::Value> indices)
    -> mlir::FailureOr<Address> {
  if (!base || !element) {
    return mlir::failure();
  }
  for (auto index : indices) {
    if (!index) {
      return mlir::failure();
    }
  }
  if (auto memref = mlir::dyn_cast<mlir::MemRefType>(base.getType())) {
    if (memref.getRank() != static_cast<int64_t>(indices.size()) ||
        memref.getElementType() != element) {
      return mlir::failure();
    }
    for (auto index : indices) {
      if (!index.getType().isIndex()) {
        return mlir::failure();
      }
    }
  } else if (!mlir::isa<mlir::LLVM::LLVMPointerType>(base.getType())) {
    return mlir::failure();
  } else {
    if (!mlir::LLVM::isCompatibleType(element)) {
      return mlir::failure();
    }
    for (auto index : indices) {
      if (!index.getType().isIndex() &&
          !mlir::isa<mlir::IntegerType>(index.getType())) {
        return mlir::failure();
      }
    }
  }
  return Address(base, element, indices);
}

auto getLLVMOffsetPointer(mlir::OpBuilder &builder, mlir::Location loc,
                          mlir::Value base, mlir::Type element,
                          llvm::ArrayRef<mlir::Value> indices) -> mlir::Value {
  if (!base || !element ||
      !mlir::isa<mlir::LLVM::LLVMPointerType>(base.getType())) {
    return {};
  }
  if (indices.empty()) {
    return base;
  }
  llvm::SmallVector<mlir::LLVM::GEPArg> converted;
  for (auto index : indices) {
    if (!index) {
      return {};
    }
    if (index.getType().isIndex()) {
      index = mlir::arith::IndexCastOp::create(builder, loc,
                                               builder.getI64Type(), index);
    }
    if (!mlir::isa<mlir::IntegerType>(index.getType())) {
      return {};
    }
    converted.push_back(index);
  }
  return mlir::LLVM::GEPOp::create(builder, loc, base.getType(), element, base,
                                   converted);
}

auto Address::materialize(mlir::OpBuilder &builder, mlir::Location loc) const
    -> mlir::Value {
  if (mlir::isa<mlir::LLVM::LLVMPointerType>(base.getType())) {
    return getLLVMOffsetPointer(builder, loc, base, element, indices);
  }
  if (indices.empty()) {
    return base;
  }
  auto type = mlir::cast<mlir::MemRefType>(base.getType());
  llvm::SmallVector<mlir::OpFoldResult> offsets, sizes, strides;
  for (auto index : indices) {
    offsets.push_back(index);
    sizes.push_back(builder.getIndexAttr(1));
    strides.push_back(builder.getIndexAttr(1));
  }
  auto result = mlir::memref::SubViewOp::inferRankReducedResultType(
      llvm::ArrayRef<int64_t>{}, type, offsets, sizes, strides);
  return mlir::memref::SubViewOp::create(builder, loc, result, base, offsets,
                                         sizes, strides);
}

auto Address::load(mlir::OpBuilder &builder, mlir::Location loc) const
    -> mlir::Value {
  if (mlir::isa<mlir::MemRefType>(base.getType())) {
    return mlir::memref::LoadOp::create(builder, loc, base, indices);
  }
  return mlir::LLVM::LoadOp::create(builder, loc, element,
                                    materialize(builder, loc));
}

auto Address::store(mlir::OpBuilder &builder, mlir::Location loc,
                    mlir::Value value) const -> mlir::LogicalResult {
  if (!value || value.getType() != element) {
    mlir::emitError(loc, "stored value does not match address element type");
    return mlir::failure();
  }
  if (mlir::isa<mlir::MemRefType>(base.getType())) {
    mlir::memref::StoreOp::create(builder, loc, value, base, indices);
  } else {
    mlir::LLVM::StoreOp::create(builder, loc, value, materialize(builder, loc));
  }
  return mlir::success();
}

auto createScalarSlot(mlir::OpBuilder &builder, mlir::Location loc,
                      mlir::Value initial) -> mlir::Value {
  if (!initial || !mlir::MemRefType::isValidElementType(initial.getType())) {
    return {};
  }
  auto slot = mlir::memref::AllocaOp::create(
      builder, loc, mlir::MemRefType::get({}, initial.getType()));
  mlir::memref::StoreOp::create(builder, loc, initial, slot);
  return slot;
}

} // namespace clang_mlir
