#ifndef CHWC_UTILS_STATE_H
#define CHWC_UTILS_STATE_H

#include "../../Converter.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Support/BackedgeBuilder.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/WithColor.h"
#include <memory>

namespace chwc::utils {

struct RegisterState {
  std::unique_ptr<circt::BackedgeBuilder> backedgeBuilder;
  llvm::DenseMap<const clang::FieldDecl *, circt::Backedge> nextBackedges;

  void init(mlir::OpBuilder &builder, mlir::Location loc) {
    if (!backedgeBuilder) {
      backedgeBuilder = std::make_unique<circt::BackedgeBuilder>(builder, loc);
    }
  }
};

inline auto emitRegister(RegisterState &state, HWModuleContext &moduleContext,
                         const clang::FieldDecl *fieldDecl,
                         mlir::OpBuilder &builder, mlir::Location loc)
    -> mlir::Value {
  HWFieldInfo &fieldInfo = moduleContext.fields[fieldDecl];

  state.init(builder, loc);

  circt::Backedge nextBackedge = state.backedgeBuilder->get(fieldInfo.type);

  state.nextBackedges[fieldDecl] = nextBackedge;

  auto reg = circt::seq::FirRegOp::create(
      builder, loc, static_cast<mlir::Value>(nextBackedge), moduleContext.clock,
      builder.getStringAttr(fieldInfo.name), moduleContext.reset,
      fieldInfo.resetValue);

  return reg.getResult();
}

inline void setRegisterNext(RegisterState &state,
                            const clang::FieldDecl *fieldDecl,
                            mlir::Value nextValue) {
  auto it = state.nextBackedges.find(fieldDecl);
  if (it == state.nextBackedges.end()) {
    llvm::WithColor::error() << "chwc: missing register next backedge\n";
    return;
  }

  it->second.setValue(nextValue);
}

} // namespace chwc::utils

#endif // CHWC_UTILS_STATE_H
