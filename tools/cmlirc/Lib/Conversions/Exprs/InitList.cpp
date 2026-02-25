#include "../../Converter.h"
#include "../Utils/Casts.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"

namespace cmlirc {

void CMLIRConverter::storeInitListValues(clang::InitListExpr *initList,
                                         mlir::Value memref) {
  mlir::OpBuilder &builder = context_manager_.Builder();
  mlir::Location loc = builder.getUnknownLoc();

  std::function<void(clang::InitListExpr *,
                     llvm::SmallVector<mlir::Value, 4> &)>
      storeValues = [&](clang::InitListExpr *list,
                        llvm::SmallVector<mlir::Value, 4> &currentIndices) {
        for (uint32_t i = 0; i < list->getNumInits(); ++i) {
          clang::Expr *init = list->getInit(i);
          mlir::Value indexVal = detail::indexConst(builder, loc, i);

          if (auto *nestedList = mlir::dyn_cast<clang::InitListExpr>(init)) {
            currentIndices.push_back(indexVal);
            storeValues(nestedList, currentIndices);
            currentIndices.pop_back();
          } else {
            mlir::Value value = generateExpr(init);
            if (!value) {
              llvm::errs() << "Failed to generate init value\n";
              continue;
            }

            llvm::SmallVector<mlir::Value, 4> fullIndices;
            fullIndices.append(currentIndices.begin(), currentIndices.end());
            fullIndices.push_back(indexVal);

            mlir::affine::AffineStoreOp::create(builder, loc, value, memref,
                                                fullIndices);
          }
        }
      };

  llvm::SmallVector<mlir::Value, 4> indices;
  storeValues(initList, indices);
}

mlir::Value
CMLIRConverter::generateInitListExpr(clang::InitListExpr *initList) {
  mlir::OpBuilder &builder = context_manager_.Builder();
  mlir::Location loc = builder.getUnknownLoc();

  clang::QualType clangType = initList->getType();
  mlir::Type mlirType = convertType(clangType);

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
