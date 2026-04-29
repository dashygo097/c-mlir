#ifndef CHWC_UTILS_STATE_H
#define CHWC_UTILS_STATE_H

#include "../../Converter.h"
#include "circt/Support/BackedgeBuilder.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"
#include "llvm/Support/WithColor.h"

namespace chwc::utils {

inline auto
emitRegister(std::unique_ptr<circt::BackedgeBuilder> &backedgeBuilder,
             llvm::DenseMap<const clang::FieldDecl *, circt::Backedge>
                 &registerNextBackedgeTable,
             mlir::Value clockValue, mlir::Value resetValue,
             mlir::OpBuilder &builder, mlir::Location loc,
             const HWFieldInfo &fieldInfo, mlir::Value resetInitValue)
    -> mlir::Value {
  if (!clockValue || !resetValue) {
    llvm::WithColor::error()
        << "chwc: register emission requires clk/rst ports\n";
    return resetInitValue;
  }

  if (!resetInitValue) {
    llvm::WithColor::error()
        << "chwc: register reset value is null: " << fieldInfo.name << "\n";
    return nullptr;
  }

  if (!backedgeBuilder) {
    llvm::WithColor::error()
        << "chwc: register emission requires active backedge builder\n";
    return resetInitValue;
  }

  circt::Backedge nextBackedge = backedgeBuilder->get(fieldInfo.type);
  registerNextBackedgeTable.insert({fieldInfo.fieldDecl, nextBackedge});

  mlir::Value nextValue = nextBackedge;

  mlir::OperationState opState(loc, "seq.firreg");
  opState.addOperands({nextValue, clockValue, resetValue, resetInitValue});
  opState.addTypes(fieldInfo.type);
  opState.addAttribute("name", builder.getStringAttr(fieldInfo.name));

  mlir::Operation *op = builder.create(opState);
  if (!op || op->getNumResults() == 0) {
    llvm::WithColor::error()
        << "chwc: failed to create seq.firreg for " << fieldInfo.name << "\n";
    return resetInitValue;
  }

  return op->getResult(0);
}

inline void
emitRegisterNextAssign(llvm::DenseMap<const clang::FieldDecl *, circt::Backedge>
                           &registerNextBackedgeTable,
                       mlir::OpBuilder &builder, mlir::Location loc,
                       const HWFieldInfo &fieldInfo, mlir::Value nextValue) {
  (void)builder;
  (void)loc;

  if (!nextValue) {
    llvm::WithColor::error()
        << "chwc: null next value for register " << fieldInfo.name << "\n";
    return;
  }

  auto backedgeIt = registerNextBackedgeTable.find(fieldInfo.fieldDecl);
  if (backedgeIt == registerNextBackedgeTable.end()) {
    llvm::WithColor::error() << "chwc: no next-state backedge for register "
                             << fieldInfo.name << "\n";
    return;
  }

  backedgeIt->second.setValue(nextValue);
}

} // namespace chwc::utils

#endif // CHWC_UTILS_STATE_H
