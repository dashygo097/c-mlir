#include "../../Converter.h"
#include "../Utils/Annotation.h"
#include "../Utils/Array.h"
#include "../Utils/Cast.h"
#include "../Utils/Expr.h"
#include "../Utils/Module.h"
#include "../Utils/State.h"
#include "../Utils/Type.h"
#include "llvm/Support/WithColor.h"

namespace chwc {

auto getFieldInfo(llvm::DenseMap<const clang::FieldDecl *, HWFieldInfo> &table,
                  const clang::FieldDecl *fieldDecl) -> HWFieldInfo * {
  if (!fieldDecl) {
    return nullptr;
  }

  auto it = table.find(fieldDecl);
  if (it == table.end()) {
    return nullptr;
  }

  return &it->second;
}

auto getFieldValueOrZero(mlir::OpBuilder &builder, mlir::Location loc,
                         const HWFieldInfo &fieldInfo, mlir::Value value)
    -> mlir::Value {
  if (value) {
    return value;
  }

  return utils::zeroFieldValue(builder, loc, fieldInfo);
}

auto CHWConverter::classifyField(clang::FieldDecl *fieldDecl)
    -> std::optional<HWFieldKind> {
  utils::ConstantArrayTypeInfo arrayInfo =
      utils::getConstantArrayTypeInfo(fieldDecl->getType());

  if (arrayInfo.isArray && arrayInfo.elementInfo.isSignal &&
      arrayInfo.elementInfo.fieldKind) {
    return arrayInfo.elementInfo.fieldKind;
  }

  utils::SignalTypeInfo typeInfo =
      utils::getSignalTypeInfo(fieldDecl->getType());

  if (typeInfo.isSignal && typeInfo.fieldKind) {
    return typeInfo.fieldKind;
  }

  std::optional<std::string> annotation = utils::getAnnotation(fieldDecl);
  if (!annotation) {
    return std::nullopt;
  }

  if (*annotation == "hw.input") {
    return HWFieldKind::Input;
  }

  if (*annotation == "hw.output") {
    return HWFieldKind::Output;
  }

  if (*annotation == "hw.reg") {
    return HWFieldKind::Reg;
  }

  if (*annotation == "hw.wire") {
    return HWFieldKind::Wire;
  }

  return std::nullopt;
}

auto CHWConverter::getAssignedField(clang::Expr *expr)
    -> const clang::FieldDecl * {
  return utils::getFieldDeclFromExpr(expr);
}

void CHWConverter::emitStateDecls() {
  mlir::OpBuilder &builder = contextManager.Builder();
  mlir::Location loc = builder.getUnknownLoc();

  for (const clang::FieldDecl *fieldDecl : hardwareFieldOrder) {
    auto fieldIt = fieldTable.find(fieldDecl);
    if (fieldIt == fieldTable.end()) {
      continue;
    }

    HWFieldInfo &fieldInfo = fieldIt->second;

    switch (fieldInfo.kind) {
    case HWFieldKind::Input: {
      currentFieldValueTable[fieldDecl] =
          utils::getInputValue(inputValueTable, builder, loc, fieldInfo);
      break;
    }

    case HWFieldKind::Output: {
      break;
    }

    case HWFieldKind::Reg: {
      if (!fieldInfo.resetValue) {
        fieldInfo.resetValue = utils::zeroFieldValue(builder, loc, fieldInfo);
      }

      mlir::Value value = utils::emitRegister(
          backedgeBuilder, registerNextBackedgeTable, clockValue, resetValue,
          builder, loc, fieldInfo, fieldInfo.resetValue);

      currentFieldValueTable[fieldDecl] = value;
      nextFieldValueTable[fieldDecl] = value;
      break;
    }

    case HWFieldKind::Wire: {
      break;
    }
    }
  }
}

auto CHWConverter::readFieldValue(const clang::FieldDecl *fieldDecl)
    -> mlir::Value {
  HWFieldInfo *fieldInfo = getFieldInfo(fieldTable, fieldDecl);
  if (!fieldInfo) {
    llvm::WithColor::error() << "chwc: unknown hardware field\n";
    return nullptr;
  }

  mlir::OpBuilder &builder = contextManager.Builder();
  mlir::Location loc = builder.getUnknownLoc();

  if (isCollectingReset && fieldInfo->kind == HWFieldKind::Reg) {
    return getFieldValueOrZero(builder, loc, *fieldInfo, fieldInfo->resetValue);
  }

  switch (fieldInfo->kind) {
  case HWFieldKind::Input: {
    return getFieldValueOrZero(builder, loc, *fieldInfo,
                               currentFieldValueTable.lookup(fieldDecl));
  }

  case HWFieldKind::Output: {
    return getFieldValueOrZero(builder, loc, *fieldInfo,
                               outputValueTable.lookup(fieldDecl));
  }

  case HWFieldKind::Reg: {
    return getFieldValueOrZero(builder, loc, *fieldInfo,
                               currentFieldValueTable.lookup(fieldDecl));
  }

  case HWFieldKind::Wire: {
    return getFieldValueOrZero(builder, loc, *fieldInfo,
                               currentFieldValueTable.lookup(fieldDecl));
  }
  }

  return nullptr;
}

auto CHWConverter::assignFieldValue(const clang::FieldDecl *fieldDecl,
                                    mlir::Value value) -> mlir::Value {
  if (!value) {
    return value;
  }

  HWFieldInfo *fieldInfo = getFieldInfo(fieldTable, fieldDecl);
  if (!fieldInfo) {
    llvm::WithColor::error() << "chwc: assignment lhs is not hardware field\n";
    return value;
  }

  mlir::OpBuilder &builder = contextManager.Builder();
  mlir::Location loc = builder.getUnknownLoc();

  if (fieldInfo->isArray) {
    if (value.getType() != fieldInfo->type) {
      llvm::WithColor::error()
          << "chwc: whole-array assignment type mismatch for "
          << fieldInfo->name << "\n";
      return value;
    }
  } else {
    value = utils::promoteValue(builder, loc, value, fieldInfo->type);
    if (!value) {
      return nullptr;
    }
  }

  if (isCollectingReset) {
    if (fieldInfo->kind != HWFieldKind::Reg) {
      llvm::WithColor::error()
          << "chwc: reset assignment only supports registers\n";
      return value;
    }

    fieldInfo->resetValue = value;
    return value;
  }

  switch (fieldInfo->kind) {
  case HWFieldKind::Input:
    llvm::WithColor::error() << "chwc: cannot assign to hardware input\n";
    break;

  case HWFieldKind::Output:
    outputValueTable[fieldDecl] = value;
    break;

  case HWFieldKind::Reg:
    nextFieldValueTable[fieldDecl] = value;
    break;

  case HWFieldKind::Wire:
    currentFieldValueTable[fieldDecl] = value;
    break;
  }

  return value;
}

auto CHWConverter::readArrayElement(const clang::FieldDecl *fieldDecl,
                                    mlir::Value index) -> mlir::Value {
  if (!index) {
    return nullptr;
  }

  HWFieldInfo *fieldInfo = getFieldInfo(fieldTable, fieldDecl);
  if (!fieldInfo) {
    llvm::WithColor::error() << "chwc: array read base is not hardware field\n";
    return nullptr;
  }

  if (!fieldInfo->isArray) {
    llvm::WithColor::error() << "chwc: array read base is not array field\n";
    return nullptr;
  }

  mlir::OpBuilder &builder = contextManager.Builder();
  mlir::Location loc = builder.getUnknownLoc();

  index = utils::coerceArrayIndex(builder, loc, index, fieldInfo->arraySize);
  if (!index) {
    return nullptr;
  }

  mlir::Value arrayValue = readFieldValue(fieldDecl);
  if (!arrayValue) {
    return nullptr;
  }

  return utils::arrayGet(builder, loc, arrayValue, index,
                         fieldInfo->elementType);
}

auto CHWConverter::assignArrayElement(const clang::FieldDecl *fieldDecl,
                                      mlir::Value index, mlir::Value value)
    -> mlir::Value {
  if (!index || !value) {
    return value;
  }

  HWFieldInfo *fieldInfo = getFieldInfo(fieldTable, fieldDecl);
  if (!fieldInfo) {
    llvm::WithColor::error()
        << "chwc: array assignment base is not hardware field\n";
    return value;
  }

  if (!fieldInfo->isArray) {
    llvm::WithColor::error()
        << "chwc: array assignment base is not array field\n";
    return value;
  }

  mlir::OpBuilder &builder = contextManager.Builder();
  mlir::Location loc = builder.getUnknownLoc();

  index = utils::coerceArrayIndex(builder, loc, index, fieldInfo->arraySize);
  if (!index) {
    return nullptr;
  }

  value = utils::promoteValue(builder, loc, value, fieldInfo->elementType);
  if (!value) {
    return nullptr;
  }

  mlir::Value oldArray;

  if (isCollectingReset) {
    if (fieldInfo->kind != HWFieldKind::Reg) {
      llvm::WithColor::error()
          << "chwc: reset array assignment only supports registers\n";
      return value;
    }

    oldArray = fieldInfo->resetValue;
    if (!oldArray) {
      oldArray = utils::zeroFieldValue(builder, loc, *fieldInfo);
    }
  } else {
    switch (fieldInfo->kind) {
    case HWFieldKind::Input:
      llvm::WithColor::error() << "chwc: cannot assign to input array\n";
      return value;

    case HWFieldKind::Output:
      oldArray = outputValueTable.lookup(fieldDecl);
      break;

    case HWFieldKind::Reg:
      oldArray = nextFieldValueTable.lookup(fieldDecl);
      break;

    case HWFieldKind::Wire:
      oldArray = currentFieldValueTable.lookup(fieldDecl);
      break;
    }

    if (!oldArray) {
      oldArray = readFieldValue(fieldDecl);
    }
  }

  if (!oldArray) {
    return value;
  }

  mlir::Value newArray =
      utils::arrayInject(builder, loc, oldArray, index, value, fieldInfo->type);

  if (!newArray) {
    return value;
  }

  if (isCollectingReset) {
    fieldInfo->resetValue = newArray;
    return value;
  }

  switch (fieldInfo->kind) {
  case HWFieldKind::Input:
    break;

  case HWFieldKind::Output:
    outputValueTable[fieldDecl] = newArray;
    break;

  case HWFieldKind::Reg:
    nextFieldValueTable[fieldDecl] = newArray;
    break;

  case HWFieldKind::Wire:
    currentFieldValueTable[fieldDecl] = newArray;
    break;
  }

  return value;
}

} // namespace chwc
