#include "../Utils/Module.h"
#include "../../Converter.h"
#include "../Utils/Constants.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/WithColor.h"

namespace chwc {

auto isMethodNamed(clang::CXXMethodDecl *methodDecl, llvm::StringRef name)
    -> bool {
  if (!methodDecl) {
    return false;
  }

  clang::DeclarationName declName = methodDecl->getDeclName();
  if (!declName.isIdentifier()) {
    return false;
  }

  clang::IdentifierInfo *identifier = declName.getAsIdentifierInfo();
  if (!identifier) {
    return false;
  }

  return identifier->getName() == name;
}

auto CHWConverter::isHardwareClass(clang::CXXRecordDecl *recordDecl) -> bool {
  if (!recordDecl) {
    return false;
  }

  for (const clang::CXXBaseSpecifier &base : recordDecl->bases()) {
    auto *baseRecord = base.getType()->getAsCXXRecordDecl();
    if (!baseRecord) {
      continue;
    }

    if (baseRecord->getNameAsString() == "Hardware") {
      return true;
    }

    if (isHardwareClass(baseRecord)) {
      return true;
    }
  }

  return false;
}

void CHWConverter::collectHardwareClass(clang::CXXRecordDecl *recordDecl) {
  hardwareFieldOrder.clear();

  fieldTable.clear();
  currentFieldValueTable.clear();
  nextFieldValueTable.clear();
  outputValueTable.clear();
  localValueTable.clear();

  resetMethod = nullptr;
  clockTickMethod = nullptr;

  for (auto *fieldDecl : recordDecl->fields()) {
    std::optional<HWFieldKind> kind = classifyField(fieldDecl);
    if (!kind) {
      continue;
    }

    mlir::Type type = convertType(fieldDecl->getType());
    if (!type) {
      llvm::WithColor::error() << "chwc: unsupported hardware field type: "
                               << fieldDecl->getType().getAsString() << "\n";
      continue;
    }

    HWFieldInfo fieldInfo;
    fieldInfo.fieldDecl = fieldDecl;
    fieldInfo.name = fieldDecl->getNameAsString();
    fieldInfo.type = type;
    fieldInfo.kind = *kind;

    hardwareFieldOrder.push_back(fieldDecl);
    fieldTable[fieldDecl] = fieldInfo;
  }

  for (auto *methodDecl : recordDecl->methods()) {
    if (isMethodNamed(methodDecl, "reset")) {
      resetMethod = methodDecl;
      continue;
    }

    if (isMethodNamed(methodDecl, "clock_tick")) {
      clockTickMethod = methodDecl;
      continue;
    }
  }

  if (!clockTickMethod) {
    llvm::WithColor::error() << "chwc: hardware class requires clock_tick()\n";
  }
}

void CHWConverter::emitHardwareClass(clang::CXXRecordDecl *recordDecl) {
  mlir::OpBuilder &builder = contextManager.Builder();
  mlir::Location loc = builder.getUnknownLoc();

  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToEnd(contextManager.Module().getBody());

  utils::beginHWModule(moduleState, builder, loc, recordDecl, fieldTable,
                       hardwareFieldOrder);

  collectResetValues();
  emitStateDecls();
  emitClockTick();

  for (const clang::FieldDecl *fieldDecl : hardwareFieldOrder) {
    auto fieldIt = fieldTable.find(fieldDecl);
    if (fieldIt == fieldTable.end()) {
      continue;
    }

    HWFieldInfo &fieldInfo = fieldIt->second;
    if (fieldInfo.kind != HWFieldKind::Output) {
      continue;
    }

    mlir::Value value = outputValueTable.lookup(fieldDecl);
    if (!value) {
      value = utils::zeroValue(builder, loc, fieldInfo.type);
    }

    utils::emitOutputAssign(moduleState, builder, loc, fieldInfo, value);
  }

  utils::endHWModule(moduleState, builder, loc);
}

} // namespace chwc
