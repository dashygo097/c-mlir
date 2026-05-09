#ifndef CHWC_UTILS_STATE_H
#define CHWC_UTILS_STATE_H

#include "../../Converter.h"
#include "./Array.h"
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

inline auto resetValueForType(mlir::OpBuilder &builder, mlir::Location loc,
                              mlir::Type type) -> mlir::Value {
  if (mlir::isa<circt::hw::ArrayType>(type)) {
    return zeroArray(builder, loc, type);
  }

  return zeroValue(builder, loc, type);
}

inline auto emitRegNext(HWModuleContext &moduleContext,
                        mlir::OpBuilder &builder, mlir::Location loc,
                        mlir::Value nextValue) -> mlir::Value {
  if (!nextValue) {
    return nullptr;
  }

  if (!moduleContext.clock || !moduleContext.reset) {
    llvm::WithColor::error() << "chwc: RegNext requires module clock/reset\n";
    return nullptr;
  }

  mlir::Value resetValue = resetValueForType(builder, loc, nextValue.getType());
  if (!resetValue) {
    llvm::WithColor::error() << "chwc: failed to create RegNext reset value\n";
    return nullptr;
  }

  std::string name = "__r_" + std::to_string(moduleContext.anonymousRegIndex++);

  auto reg = circt::seq::FirRegOp::create(
      builder, loc, nextValue, moduleContext.clock, builder.getStringAttr(name),
      moduleContext.reset, resetValue);

  return reg.getResult();
}

inline auto emitDelay(HWModuleContext &moduleContext, mlir::OpBuilder &builder,
                      mlir::Location loc, mlir::Value value, unsigned cycles)
    -> mlir::Value {
  if (!value) {
    return nullptr;
  }

  if (cycles == 0) {
    llvm::WithColor::error() << "chwc: Delay cycle count must be >= 1\n";
    return nullptr;
  }

  mlir::Value delayed = value;

  for (unsigned i = 0; i < cycles; ++i) {
    delayed = emitRegNext(moduleContext, builder, loc, delayed);
    if (!delayed) {
      return nullptr;
    }
  }

  return delayed;
}

} // namespace chwc::utils

#endif // CHWC_UTILS_STATE_H
