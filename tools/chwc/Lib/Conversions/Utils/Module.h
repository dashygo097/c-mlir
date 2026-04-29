#ifndef CHWC_UTILS_MODULE_H
#define CHWC_UTILS_MODULE_H

#include "../../Converter.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/PortImplementation.h"
#include "circt/Dialect/Seq/SeqTypes.h"
#include "circt/Support/BackedgeBuilder.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/WithColor.h"

namespace chwc::utils {

inline void beginHWModule(
    circt::hw::HWModuleOp &moduleOp, mlir::Value &clockValue,
    mlir::Value &resetValue,
    std::unique_ptr<circt::BackedgeBuilder> &backedgeBuilder,
    llvm::DenseMap<const clang::FieldDecl *, mlir::Value> &inputValueTable,
    llvm::SmallVectorImpl<mlir::Value> &outputValues,
    llvm::DenseMap<const clang::FieldDecl *, circt::Backedge>
        &registerNextBackedgeTable,
    mlir::OpBuilder &builder, mlir::Location loc,
    clang::CXXRecordDecl *recordDecl,
    llvm::DenseMap<const clang::FieldDecl *, HWFieldInfo> &fieldTable,
    llvm::ArrayRef<const clang::FieldDecl *> fieldOrder) {
  moduleOp = nullptr;
  clockValue = nullptr;
  resetValue = nullptr;

  if (backedgeBuilder) {
    backedgeBuilder->abandon();
  }
  backedgeBuilder.reset();

  inputValueTable.clear();
  outputValues.clear();
  registerNextBackedgeTable.clear();

  llvm::SmallVector<circt::hw::PortInfo, 8> ports;

  size_t inputArgNo = 0;
  size_t outputArgNo = 0;

  mlir::Type clkType = circt::seq::ClockType::get(builder.getContext());
  if (!clkType) {
    return;
  }

  {
    circt::hw::PortInfo port;
    port.name = builder.getStringAttr("clk");
    port.type = clkType;
    port.dir = circt::hw::ModulePort::Direction::Input;
    port.argNum = inputArgNo++;
    port.loc = loc;
    ports.push_back(port);
  }

  {
    circt::hw::PortInfo port;
    port.name = builder.getStringAttr("rst");
    port.type = builder.getI1Type();
    port.dir = circt::hw::ModulePort::Direction::Input;
    port.argNum = inputArgNo++;
    port.loc = loc;
    ports.push_back(port);
  }

  for (const clang::FieldDecl *fieldDecl : fieldOrder) {
    auto fieldIt = fieldTable.find(fieldDecl);
    if (fieldIt == fieldTable.end()) {
      continue;
    }

    HWFieldInfo &fieldInfo = fieldIt->second;

    if (fieldInfo.kind == HWFieldKind::Input) {
      circt::hw::PortInfo port;
      port.name = builder.getStringAttr(fieldInfo.name);
      port.type = fieldInfo.type;
      port.dir = circt::hw::ModulePort::Direction::Input;
      port.argNum = inputArgNo++;
      port.loc = loc;
      ports.push_back(port);
      continue;
    }

    if (fieldInfo.kind == HWFieldKind::Output) {
      circt::hw::PortInfo port;
      port.name = builder.getStringAttr(fieldInfo.name);
      port.type = fieldInfo.type;
      port.dir = circt::hw::ModulePort::Direction::Output;
      port.argNum = outputArgNo++;
      port.loc = loc;
      ports.push_back(port);
      continue;
    }
  }

  moduleOp = circt::hw::HWModuleOp::create(
      builder, loc, builder.getStringAttr(recordDecl->getNameAsString()),
      ports);

  mlir::Block *bodyBlock = moduleOp.getBodyBlock();

  if (!bodyBlock->empty()) {
    if (auto outputOp =
            mlir::dyn_cast<circt::hw::OutputOp>(bodyBlock->back())) {
      outputOp.erase();
    }
  }

  if (bodyBlock->getNumArguments() < 2) {
    llvm::WithColor::error()
        << "chwc: internal error: clk/rst block args missing\n";
    return;
  }

  clockValue = bodyBlock->getArgument(0);
  resetValue = bodyBlock->getArgument(1);

  size_t argIndex = 2;

  for (const clang::FieldDecl *fieldDecl : fieldOrder) {
    auto fieldIt = fieldTable.find(fieldDecl);
    if (fieldIt == fieldTable.end()) {
      continue;
    }

    HWFieldInfo &fieldInfo = fieldIt->second;
    if (fieldInfo.kind != HWFieldKind::Input) {
      continue;
    }

    if (argIndex >= bodyBlock->getNumArguments()) {
      llvm::WithColor::error()
          << "chwc: internal error: hw.module input block arg missing\n";
      continue;
    }

    inputValueTable[fieldDecl] = bodyBlock->getArgument(argIndex++);
  }

  builder.setInsertionPointToEnd(bodyBlock);

  backedgeBuilder = std::make_unique<circt::BackedgeBuilder>(builder, loc);
}

inline void endHWModule(
    circt::hw::HWModuleOp &moduleOp, mlir::Value &clockValue,
    mlir::Value &resetValue,
    std::unique_ptr<circt::BackedgeBuilder> &backedgeBuilder,
    llvm::DenseMap<const clang::FieldDecl *, mlir::Value> &inputValueTable,
    llvm::SmallVectorImpl<mlir::Value> &outputValues,
    llvm::DenseMap<const clang::FieldDecl *, circt::Backedge>
        &registerNextBackedgeTable,
    mlir::OpBuilder &builder, mlir::Location loc) {
  if (!moduleOp) {
    return;
  }

  if (backedgeBuilder) {
    if (mlir::failed(backedgeBuilder->clearOrEmitError())) {
      llvm::WithColor::error()
          << "chwc: unresolved register next-state backedge\n";
    }
    backedgeBuilder.reset();
  }

  circt::hw::OutputOp::create(builder, loc, outputValues);

  builder.setInsertionPointAfter(moduleOp);

  moduleOp = nullptr;
  clockValue = nullptr;
  resetValue = nullptr;
  inputValueTable.clear();
  outputValues.clear();
  registerNextBackedgeTable.clear();
}

inline auto getInputValue(
    llvm::DenseMap<const clang::FieldDecl *, mlir::Value> &inputValueTable,
    mlir::OpBuilder &builder, mlir::Location loc, const HWFieldInfo &fieldInfo)
    -> mlir::Value {
  (void)builder;
  (void)loc;

  mlir::Value value = inputValueTable.lookup(fieldInfo.fieldDecl);
  if (!value) {
    llvm::WithColor::error()
        << "chwc: input port value is not wired: " << fieldInfo.name << "\n";
  }

  return value;
}

inline void emitOutputAssign(llvm::SmallVectorImpl<mlir::Value> &outputValues,
                             mlir::OpBuilder &builder, mlir::Location loc,
                             const HWFieldInfo &fieldInfo, mlir::Value value) {
  (void)builder;
  (void)loc;

  if (!value) {
    llvm::WithColor::error()
        << "chwc: null output value for " << fieldInfo.name << "\n";
    return;
  }

  outputValues.push_back(value);
}

} // namespace chwc::utils

#endif // CHWC_UTILS_MODULE_H
