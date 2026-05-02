#include "../Utils/Module.h"
#include "../../Converter.h"
#include "../Utils/Annotation.h"
#include "../Utils/Constants.h"
#include "llvm/Support/WithColor.h"

namespace chwc {

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
  clearHardwareState();

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
    if (utils::isResetMethod(methodDecl)) {
      resetMethods.push_back(methodDecl);
      continue;
    }

    if (utils::isClockTickMethod(methodDecl)) {
      clockTickMethods.push_back(methodDecl);
      continue;
    }
  }

  if (clockTickMethods.empty()) {
    llvm::WithColor::error()
        << "chwc: hardware class requires at least one "
           "[[clang::annotate(\"hw.clock_tick\")]] method\n";
  }
}

void CHWConverter::emitHardwareClass(clang::CXXRecordDecl *recordDecl) {
  mlir::OpBuilder &builder = contextManager.Builder();
  mlir::Location loc = builder.getUnknownLoc();

  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToEnd(contextManager.Module().getBody());

  utils::beginHWModule(currentModuleOp, clockValue, resetValue, backedgeBuilder,
                       inputValueTable, outputValues, registerNextBackedgeTable,
                       builder, loc, recordDecl, fieldTable,
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

    utils::emitOutputAssign(outputValues, builder, loc, fieldInfo, value);
  }

  utils::endHWModule(currentModuleOp, clockValue, resetValue, backedgeBuilder,
                     inputValueTable, outputValues, registerNextBackedgeTable,
                     builder, loc);
}

} // namespace chwc
