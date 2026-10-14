#include "../Lowering.h"
#include "clang-mlir/Constants.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "llvm/Support/WithColor.h"

namespace cmlir::frontend {

auto ASTLowering::storeInitListValues(clang::InitListExpr *list,
                                      mlir::Value memref) -> bool {
  llvm::SmallVector<mlir::Value, 4> indices;
  return storeInitListValues(list, memref, indices);
}

auto ASTLowering::storeInitListValues(
    clang::InitListExpr *list, mlir::Value memref,
    llvm::SmallVector<mlir::Value, 4> &indices) -> bool {
  auto &builder = context.builder();
  auto loc = context.source().location(list->getExprLoc());
  auto type = mlir::cast<mlir::MemRefType>(memref.getType());
  for (unsigned i = 0; i < list->getNumInits(); ++i) {
    auto *initializer = list->getInit(i);
    indices.push_back(clang_mlir::indexConst(builder, loc, i));
    if (auto *nested = llvm::dyn_cast<clang::InitListExpr>(initializer)) {
      if (!storeInitListValues(nested, memref, indices)) {
        return false;
      }
    } else {
      auto value = generateExpr(initializer);
      if (!value) {
        return false;
      }
      auto address =
          clang_mlir::Address::get(memref, type.getElementType(), indices);
      if (mlir::failed(address) ||
          mlir::failed(address->store(builder, loc, value))) {
        context.source().error(initializer->getExprLoc(),
                               "invalid array initializer storage");
        return false;
      }
    }
    indices.pop_back();
  }
  return true;
}

auto ASTLowering::generateInitListExpr(clang::InitListExpr *initList)
    -> mlir::Value {
  mlir::OpBuilder &builder = context.builder();
  mlir::Location loc = builder.getUnknownLoc();

  clang::QualType clangType = initList->getType();
  mlir::Type mlirType = convertType(clangType);

  auto memrefType = mlir::dyn_cast<mlir::MemRefType>(mlirType);
  if (!memrefType) {
    llvm::WithColor::error()
        << "clang-to-mlir: InitListExpr must have memref type\n";
    return nullptr;
  }

  auto allocaOp = mlir::memref::AllocaOp::create(builder, loc, memrefType);
  mlir::Value memref = allocaOp.getResult();

  if (!storeInitListValues(initList, memref)) {
    return {};
  }

  return memref;
}

} // namespace cmlir::frontend
