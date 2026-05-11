#include "../../Converter.h"
#include "../Utils/Annotation.h"
#include "../Utils/Array.h"
#include "../Utils/Constant.h"
#include "../Utils/Module.h"
#include "../Utils/State.h"
#include "../Utils/Template.h"
#include "llvm/Support/WithColor.h"

namespace chwc {

auto isModuleClassImpl(clang::CXXRecordDecl *recordDecl) -> bool {
  if (!recordDecl) {
    return false;
  }

  for (const clang::CXXBaseSpecifier &base : recordDecl->bases()) {
    auto *baseRecord = base.getType()->getAsCXXRecordDecl();
    if (!baseRecord) {
      continue;
    }

    if (baseRecord->getNameAsString() == "Module") {
      return true;
    }

    if (isModuleClassImpl(baseRecord)) {
      return true;
    }
  }

  return false;
}

auto makeIndexConst(mlir::OpBuilder &builder, mlir::Location loc,
                    uint64_t value) -> mlir::Value {
  return utils::intConst(builder, loc, builder.getIntegerType(32), value);
}

auto buildArrayResetValue(mlir::OpBuilder &builder, mlir::Location loc,
                          HWFieldInfo &fieldInfo) -> mlir::Value {
  mlir::Value resetArray = utils::zeroArray(builder, loc, fieldInfo.type);
  if (!resetArray) {
    return nullptr;
  }

  for (uint64_t i = 0; i < fieldInfo.arraySize; ++i) {
    int64_t initValue = fieldInfo.regInitValue;

    if (!fieldInfo.regInitValues.empty()) {
      initValue = 0;

      if (i < fieldInfo.regInitValues.size()) {
        initValue = fieldInfo.regInitValues[i];
      }
    }

    if (initValue == 0) {
      continue;
    }

    mlir::Value index = makeIndexConst(builder, loc, i);
    mlir::Value elem =
        utils::intConst(builder, loc, fieldInfo.elementType, initValue);

    if (!index || !elem) {
      return nullptr;
    }

    resetArray = utils::arrayInject(builder, loc, resetArray, index, elem);
    if (!resetArray) {
      return nullptr;
    }
  }

  return resetArray;
}

auto buildDefaultResetValue(mlir::OpBuilder &builder, mlir::Location loc,
                            HWFieldInfo &fieldInfo) -> mlir::Value {
  if (fieldInfo.isArray) {
    return buildArrayResetValue(builder, loc, fieldInfo);
  }

  return utils::intConst(builder, loc, fieldInfo.type, fieldInfo.regInitValue);
}

auto CHWConverter::TraverseCXXRecordDecl(clang::CXXRecordDecl *recordDecl)
    -> bool {
  if (!recordDecl || !recordDecl->isThisDeclarationADefinition()) {
    return true;
  }

  if (!isModuleClassImpl(recordDecl)) {
    return true;
  }

  moduleContext.clear();
  moduleContext.recordDecl = recordDecl;

  mlir::OpBuilder &builder = contextManager.Builder();
  mlir::Location loc = builder.getUnknownLoc();

  utils::collectTemplateParameters(moduleContext, builder, recordDecl);

  for (clang::FieldDecl *fieldDecl : recordDecl->fields()) {
    TraverseFieldDecl(fieldDecl);
  }

  for (clang::CXXMethodDecl *methodDecl : recordDecl->methods()) {
    if (utils::isResetMethod(methodDecl)) {
      moduleContext.resetMethods.push_back(methodDecl);
      continue;
    }

    if (utils::isClockTickMethod(methodDecl)) {
      moduleContext.clockMethods.push_back(methodDecl);
      continue;
    }
  }

  if (moduleContext.clockMethods.empty()) {
    llvm::WithColor::error() << "chwc: module class requires HW_CLOCK_TICK\n";
  }

  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToEnd(contextManager.Module().getBody());

  utils::beginHWModule(moduleContext, builder, loc, recordDecl);

  for (const clang::FieldDecl *fieldDecl : moduleContext.fieldOrder) {
    HWFieldInfo &fieldInfo = moduleContext.fields[fieldDecl];

    if (fieldInfo.kind != HWFieldKind::Reg) {
      continue;
    }

    if (!fieldInfo.isArray) {
      continue;
    }

    fieldInfo.resetValue = buildDefaultResetValue(builder, loc, fieldInfo);
  }

  emitMode = HWEmitMode::Reset;
  functionStack.emplace_back();

  for (const clang::CXXMethodDecl *methodDecl : moduleContext.resetMethods) {
    if (methodDecl && methodDecl->hasBody()) {
      TraverseStmt(methodDecl->getBody());
    }
  }

  functionStack.pop_back();
  emitMode = HWEmitMode::Normal;

  utils::RegisterState registerState;

  for (const clang::FieldDecl *fieldDecl : moduleContext.fieldOrder) {
    HWFieldInfo &fieldInfo = moduleContext.fields[fieldDecl];

    switch (fieldInfo.kind) {
    case HWFieldKind::Input:
      moduleContext.currentValues[fieldDecl] =
          utils::getInputValue(moduleContext, fieldDecl, builder, loc);
      break;

    case HWFieldKind::Output:
      break;

    case HWFieldKind::Wire:
      if (fieldInfo.isArray) {
        moduleContext.currentValues[fieldDecl] =
            utils::zeroArray(builder, loc, fieldInfo.type);
      } else {
        moduleContext.currentValues[fieldDecl] =
            utils::zeroValue(builder, loc, fieldInfo.type);
      }
      break;

    case HWFieldKind::Reg: {
      if (!fieldInfo.resetValue) {
        fieldInfo.resetValue = buildDefaultResetValue(builder, loc, fieldInfo);
      }

      mlir::Value reg = utils::emitRegister(registerState, moduleContext,
                                            fieldDecl, builder, loc);

      moduleContext.currentValues[fieldDecl] = reg;
      moduleContext.nextValues[fieldDecl] = reg;
      break;
    }
    }
  }

  functionStack.emplace_back();

  for (const clang::CXXMethodDecl *methodDecl : moduleContext.clockMethods) {
    if (methodDecl && methodDecl->hasBody()) {
      TraverseStmt(methodDecl->getBody());
    }
  }

  functionStack.pop_back();

  for (const clang::FieldDecl *fieldDecl : moduleContext.fieldOrder) {
    HWFieldInfo &fieldInfo = moduleContext.fields[fieldDecl];

    if (fieldInfo.kind != HWFieldKind::Reg) {
      continue;
    }

    mlir::Value nextValue = moduleContext.nextValues.lookup(fieldDecl);
    if (!nextValue) {
      nextValue = moduleContext.currentValues.lookup(fieldDecl);
    }

    if (!nextValue) {
      llvm::WithColor::error()
          << "chwc: missing next value for register: " << fieldInfo.name
          << "\n";
      continue;
    }

    utils::setRegisterNext(registerState, fieldDecl, nextValue);
  }

  for (const clang::FieldDecl *fieldDecl : moduleContext.fieldOrder) {
    HWFieldInfo &fieldInfo = moduleContext.fields[fieldDecl];

    if (fieldInfo.kind != HWFieldKind::Output) {
      continue;
    }

    mlir::Value value = moduleContext.outputValues.lookup(fieldDecl);
    if (!value) {
      value = fieldInfo.isArray
                  ? utils::zeroArray(builder, loc, fieldInfo.type)
                  : utils::zeroValue(builder, loc, fieldInfo.type);
    }

    utils::emitOutputValue(moduleContext, fieldDecl, value);
  }

  utils::endHWModule(moduleContext, builder, loc);
  return true;
}

} // namespace chwc
