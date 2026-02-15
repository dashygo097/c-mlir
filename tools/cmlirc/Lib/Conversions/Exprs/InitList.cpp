#include "../../Converter.h"
#include "../Types/Types.h"

namespace cmlirc {

mlir::Value
CMLIRConverter::generateInitListExpr(clang::InitListExpr *initList) {
  mlir::OpBuilder &builder = context_manager_.Builder();
  mlir::Location loc = builder.getUnknownLoc();

  clang::QualType clangType = initList->getType();
  mlir::Type mlirType = convertType(builder, clangType);

  auto memrefType = mlir::dyn_cast<mlir::MemRefType>(mlirType);
  if (!memrefType) {
    llvm::errs() << "InitListExpr must have memref type\n";
    return nullptr;
  }

  auto allocaOp = mlir::memref::AllocaOp::create(builder, loc, memrefType);
  mlir::Value memref = allocaOp.getResult();

  storeInitListValues(initList, memref);

  return memref;
}

} // namespace cmlirc
