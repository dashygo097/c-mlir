#include "../../Converter.h"
#include "../Utils/Annotation.h"
#include "../Utils/Array.h"
#include "../Utils/Constant.h"
#include "../Utils/Module.h"
#include "../Utils/State.h"
#include "circt/Dialect/HW/HWAttributes.h"
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

auto getTemplateIntegerDefaultAttr(mlir::OpBuilder &builder,
                                   clang::NonTypeTemplateParmDecl *param)
    -> mlir::Attribute {
  if (!param || !param->hasDefaultArgument()) {
    return {};
  }

  mlir::Type paramType = builder.getIntegerType(32);

  clang::TemplateArgumentLoc defaultArgLoc = param->getDefaultArgument();
  const clang::TemplateArgument &defaultArg = defaultArgLoc.getArgument();

  if (defaultArg.getKind() == clang::TemplateArgument::Integral) {
    return mlir::IntegerAttr::get(paramType,
                                  defaultArg.getAsIntegral().getSExtValue());
  }

  clang::Expr *defaultExpr = defaultArgLoc.getSourceExpression();
  if (!defaultExpr) {
    llvm::WithColor::error() << "chwc: unsupported template parameter default: "
                             << param->getNameAsString() << "\n";
    return {};
  }

  defaultExpr = defaultExpr->IgnoreParenImpCasts();

  auto *intLit = llvm::dyn_cast<clang::IntegerLiteral>(defaultExpr);
  if (!intLit) {
    llvm::WithColor::error()
        << "chwc: template parameter default must be integer literal: "
        << param->getNameAsString() << "\n";
    return {};
  }

  return mlir::IntegerAttr::get(paramType, intLit->getValue().getSExtValue());
}

void collectTemplateParameters(CHWConverter &converter,
                               HWModuleContext &moduleContext,
                               mlir::OpBuilder &builder,
                               clang::CXXRecordDecl *recordDecl) {
  clang::ClassTemplateDecl *classTemplate =
      recordDecl->getDescribedClassTemplate();

  if (!classTemplate) {
    return;
  }

  clang::TemplateParameterList *params = classTemplate->getTemplateParameters();

  for (clang::NamedDecl *paramDecl : *params) {
    auto *nttp = llvm::dyn_cast<clang::NonTypeTemplateParmDecl>(paramDecl);
    if (!nttp) {
      llvm::WithColor::error()
          << "chwc: only non-type integer template parameters are supported\n";
      continue;
    }

    if (!nttp->getType()->isIntegerType()) {
      llvm::WithColor::error()
          << "chwc: template module parameter must be integer: "
          << nttp->getNameAsString() << "\n";
      continue;
    }

    mlir::Type paramType = builder.getIntegerType(32);
    mlir::StringAttr nameAttr = builder.getStringAttr(nttp->getNameAsString());
    mlir::Attribute defaultAttr = getTemplateIntegerDefaultAttr(builder, nttp);

    mlir::Attribute paramDeclAttr = circt::hw::ParamDeclAttr::get(
        builder.getContext(), nameAttr, paramType, defaultAttr);

    mlir::TypedAttr paramRefAttr = circt::hw::ParamDeclRefAttr::get(
        builder.getContext(), nameAttr, paramType);

    moduleContext.parameters.push_back(paramDeclAttr);
    moduleContext.parameterRefs[nttp->getNameAsString()] = paramRefAttr;
  }

  (void)converter;
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

  collectTemplateParameters(*this, moduleContext, builder, recordDecl);

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

    if (fieldInfo.isArray) {
      fieldInfo.resetValue = utils::zeroArray(builder, loc, fieldInfo.type);
    } else {
      fieldInfo.resetValue = utils::zeroValue(builder, loc, fieldInfo.type);
    }
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
        fieldInfo.resetValue =
            fieldInfo.isArray ? utils::zeroArray(builder, loc, fieldInfo.type)
                              : utils::zeroValue(builder, loc, fieldInfo.type);
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
