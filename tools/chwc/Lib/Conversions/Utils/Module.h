#ifndef CHWC_UTILS_MODULE_H
#define CHWC_UTILS_MODULE_H

#include "../../Converter.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/PortImplementation.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/WithColor.h"

namespace chwc::utils {

inline void
beginHWModule(HWModuleState &state, mlir::OpBuilder &builder,
              mlir::Location loc, clang::CXXRecordDecl *recordDecl,
              llvm::DenseMap<const clang::FieldDecl *, HWFieldInfo> &fieldTable,
              llvm::ArrayRef<const clang::FieldDecl *> fieldOrder) {
  state.clear();

  llvm::SmallVector<circt::hw::PortInfo, 8> ports;

  size_t inputArgNo = 0;
  size_t outputArgNo = 0;

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

  state.moduleOp = circt::hw::HWModuleOp::create(
      builder, loc, builder.getStringAttr(recordDecl->getNameAsString()),
      ports);

  mlir::Block *bodyBlock = state.moduleOp.getBodyBlock();

  if (!bodyBlock->empty()) {
    if (auto outputOp =
            mlir::dyn_cast<circt::hw::OutputOp>(bodyBlock->back())) {
      outputOp.erase();
    }
  }

  size_t argIndex = 0;

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

    state.inputValueTable[fieldDecl] = bodyBlock->getArgument(argIndex++);
  }

  builder.setInsertionPointToEnd(bodyBlock);
}

inline void endHWModule(HWModuleState &state, mlir::OpBuilder &builder,
                        mlir::Location loc) {
  if (!state.moduleOp) {
    return;
  }

  circt::hw::OutputOp::create(builder, loc, state.outputValues);

  builder.setInsertionPointAfter(state.moduleOp);

  state.clear();
}

inline auto getInputValue(HWModuleState &state, mlir::OpBuilder &builder,
                          mlir::Location loc, const HWFieldInfo &fieldInfo)
    -> mlir::Value {
  (void)builder;
  (void)loc;

  mlir::Value value = state.inputValueTable.lookup(fieldInfo.fieldDecl);
  if (!value) {
    llvm::WithColor::error()
        << "chwc: input port value is not wired: " << fieldInfo.name << "\n";
  }

  return value;
}

inline void emitOutputAssign(HWModuleState &state, mlir::OpBuilder &builder,
                             mlir::Location loc, const HWFieldInfo &fieldInfo,
                             mlir::Value value) {
  (void)builder;
  (void)loc;

  if (!value) {
    llvm::WithColor::error()
        << "chwc: null output value for " << fieldInfo.name << "\n";
    return;
  }

  state.outputValues.push_back(value);
}

} // namespace chwc::utils

#endif // CHWC_UTILS_MODULE_H
