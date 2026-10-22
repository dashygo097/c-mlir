#ifndef CLANG_MLIR_ADDRESS_H
#define CLANG_MLIR_ADDRESS_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SmallVector.h"

namespace clang_mlir {

// An address is a complete, validated access description. It owns no IR and
// must not outlive its module. No AST node classification is needed to use it.
class Address {
public:
  static auto get(mlir::Value base, mlir::Type element,
                  llvm::ArrayRef<mlir::Value> indices = {})
      -> mlir::FailureOr<Address>;
  auto load(mlir::OpBuilder &, mlir::Location) const -> mlir::Value;
  auto store(mlir::OpBuilder &, mlir::Location, mlir::Value) const
      -> mlir::LogicalResult;
  auto materialize(mlir::OpBuilder &, mlir::Location) const -> mlir::Value;

  auto elementType() const -> mlir::Type { return element; }

private:
  Address(mlir::Value base, mlir::Type element,
          llvm::ArrayRef<mlir::Value> indices)
      : base(base), element(element), indices(indices) {}

  mlir::Value base;
  mlir::Type element;
  llvm::SmallVector<mlir::Value, 4> indices;
};

// Select storage from the lowered object type, independently of AST syntax.
auto createLocalStorage(mlir::OpBuilder &, mlir::Location, mlir::Type,
                        mlir::ValueRange dynamicSizes = {}) -> mlir::Value;
auto createLocalSlot(mlir::OpBuilder &, mlir::Location, mlir::Value initial)
    -> mlir::Value;
auto getLLVMOffsetPointer(mlir::OpBuilder &, mlir::Location, mlir::Value base,
                          mlir::Type element,
                          llvm::ArrayRef<mlir::Value> indices) -> mlir::Value;

} // namespace clang_mlir

#endif // CLANG_MLIR_ADDRESS_H
