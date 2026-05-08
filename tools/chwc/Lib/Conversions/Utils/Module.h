#ifndef CHWC_UTILS_MODULE_H
#define CHWC_UTILS_MODULE_H

#include "../../Converter.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Dialect/Seq/SeqTypes.h"
#include "mlir/IR/Builders.h"
#include "llvm/Support/WithColor.h"

namespace chwc::utils {

inline auto makeClockType(mlir::MLIRContext *context) -> mlir::Type {
  return circt::seq::ClockType::get(context);
}

inline auto makePortInfo(mlir::StringAttr name, mlir::Type type,
                         circt::hw::ModulePort::Direction direction,
                         size_t argNum, mlir::Location loc)
    -> circt::hw::PortInfo {
  return circt::hw::PortInfo{circt::hw::ModulePort{name, type, direction},
                             argNum, mlir::DictionaryAttr{}, loc};
}

inline void removeDefaultOutputTerminator(circt::hw::HWModuleOp moduleOp) {
  mlir::Block *body = moduleOp.getBodyBlock();
  if (!body || body->empty()) {
    return;
  }

  mlir::Operation &lastOp = body->back();
  if (mlir::isa<circt::hw::OutputOp>(lastOp)) {
    lastOp.erase();
  }
}

inline void beginHWModule(HWModuleContext &moduleContext,
                          mlir::OpBuilder &builder, mlir::Location loc,
                          clang::CXXRecordDecl *recordDecl) {
  llvm::SmallVector<circt::hw::PortInfo, 16> ports;

  size_t inputArgNum = 0;
  size_t outputArgNum = 0;

  ports.push_back(makePortInfo(
      builder.getStringAttr("clk"), makeClockType(builder.getContext()),
      circt::hw::ModulePort::Direction::Input, inputArgNum++, loc));

  ports.push_back(makePortInfo(
      builder.getStringAttr("rst"), builder.getI1Type(),
      circt::hw::ModulePort::Direction::Input, inputArgNum++, loc));

  for (const clang::FieldDecl *fieldDecl : moduleContext.fieldOrder) {
    HWFieldInfo &fieldInfo = moduleContext.fields[fieldDecl];

    if (fieldInfo.kind == HWFieldKind::Input) {
      ports.push_back(makePortInfo(
          builder.getStringAttr(fieldInfo.name), fieldInfo.type,
          circt::hw::ModulePort::Direction::Input, inputArgNum++, loc));
      continue;
    }

    if (fieldInfo.kind == HWFieldKind::Output) {
      ports.push_back(makePortInfo(
          builder.getStringAttr(fieldInfo.name), fieldInfo.type,
          circt::hw::ModulePort::Direction::Output, outputArgNum++, loc));
      continue;
    }
  }

  moduleContext.moduleOp = circt::hw::HWModuleOp::create(
      builder, loc, builder.getStringAttr(recordDecl->getNameAsString()), ports,
      mlir::ArrayAttr::get(builder.getContext(), moduleContext.parameters));

  removeDefaultOutputTerminator(moduleContext.moduleOp);

  mlir::Block *body = moduleContext.moduleOp.getBodyBlock();
  builder.setInsertionPointToEnd(body);

  unsigned blockArgIndex = 0;
  moduleContext.clock = body->getArgument(blockArgIndex++);
  moduleContext.reset = body->getArgument(blockArgIndex++);

  for (const clang::FieldDecl *fieldDecl : moduleContext.fieldOrder) {
    HWFieldInfo &fieldInfo = moduleContext.fields[fieldDecl];

    if (fieldInfo.kind != HWFieldKind::Input) {
      continue;
    }

    moduleContext.currentValues[fieldDecl] = body->getArgument(blockArgIndex++);
  }
}

inline auto getInputValue(HWModuleContext &moduleContext,
                          const clang::FieldDecl *fieldDecl,
                          mlir::OpBuilder &builder, mlir::Location loc)
    -> mlir::Value {
  (void)builder;
  (void)loc;

  mlir::Value value = moduleContext.currentValues.lookup(fieldDecl);
  if (!value) {
    llvm::WithColor::error() << "chwc: input value is not available\n";
  }

  return value;
}

inline void emitOutputValue(HWModuleContext &moduleContext,
                            const clang::FieldDecl *fieldDecl,
                            mlir::Value value) {
  moduleContext.outputValues[fieldDecl] = value;
}

inline void endHWModule(HWModuleContext &moduleContext,
                        mlir::OpBuilder &builder, mlir::Location loc) {
  llvm::SmallVector<mlir::Value, 8> outputValues;

  for (const clang::FieldDecl *fieldDecl : moduleContext.fieldOrder) {
    HWFieldInfo &fieldInfo = moduleContext.fields[fieldDecl];

    if (fieldInfo.kind != HWFieldKind::Output) {
      continue;
    }

    mlir::Value value = moduleContext.outputValues.lookup(fieldDecl);
    if (!value) {
      llvm::WithColor::error()
          << "chwc: output value is not assigned: " << fieldInfo.name << "\n";
      continue;
    }

    outputValues.push_back(value);
  }

  builder.setInsertionPointToEnd(moduleContext.moduleOp.getBodyBlock());
  circt::hw::OutputOp::create(builder, loc, outputValues);
}

} // namespace chwc::utils

#endif // CHWC_UTILS_MODULE_H
