#include "../Utils/Module.h"
#include "../../Converter.h"
#include "../Utils/Annotation.h"
#include "../Utils/Array.h"
#include "../Utils/Type.h"
#include "circt/Dialect/HW/HWAttributes.h"
#include "llvm/Support/Casting.h"
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

  if (clang::ClassTemplateDecl *templateDecl =
          recordDecl->getDescribedClassTemplate()) {
    if (clang::TemplateParameterList *params =
            templateDecl->getTemplateParameters()) {
      for (clang::NamedDecl *param : *params) {
        auto *nonTypeParam =
            llvm::dyn_cast_or_null<clang::NonTypeTemplateParmDecl>(param);

        if (!nonTypeParam) {
          llvm::WithColor::error()
              << "chwc: only non-type integer template parameters are "
                 "supported: "
              << param->getNameAsString() << "\n";
          continue;
        }

        TraverseNonTypeTemplateParmDecl(nonTypeParam);
      }
    }
  }

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

    utils::ConstantArrayTypeInfo arrayInfo =
        utils::getConstantArrayTypeInfo(fieldDecl->getType());

    fieldInfo.isArray = arrayInfo.isArray;
    if (arrayInfo.isArray) {
      fieldInfo.arraySize = arrayInfo.size;
      fieldInfo.elementType = convertType(arrayInfo.elementType);
      if (!fieldInfo.elementType) {
        llvm::WithColor::error()
            << "chwc: unsupported hardware array element type: "
            << arrayInfo.elementType.getAsString() << "\n";
        continue;
      }
    } else {
      fieldInfo.elementType = fieldInfo.type;
    }

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
                       builder, loc, recordDecl, paramTable, hardwareParamOrder,
                       fieldTable, hardwareFieldOrder);

  for (const clang::NonTypeTemplateParmDecl *paramDecl : hardwareParamOrder) {
    auto paramIt = paramTable.find(paramDecl);
    if (paramIt == paramTable.end()) {
      continue;
    }

    const HWParamInfo &paramInfo = paramIt->second;

    mlir::Attribute refAttr = circt::hw::ParamDeclRefAttr::get(
        builder.getContext(), builder.getStringAttr(paramInfo.name),
        paramInfo.type);

    mlir::OperationState opState(loc, "hw.param.value");
    opState.addTypes(paramInfo.type);
    opState.addAttribute("value", refAttr);

    mlir::Operation *op = builder.create(opState);
    if (!op || op->getNumResults() == 0) {
      llvm::WithColor::error() << "chwc: failed to create hw.param.value for "
                               << paramInfo.name << "\n";
      continue;
    }

    paramValueTable[paramDecl] = op->getResult(0);
  }

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
      value = utils::zeroFieldValue(builder, loc, fieldInfo);
    }

    utils::emitOutputAssign(outputValues, builder, loc, fieldInfo, value);
  }

  utils::endHWModule(currentModuleOp, clockValue, resetValue, backedgeBuilder,
                     inputValueTable, outputValues, registerNextBackedgeTable,
                     builder, loc);
}

} // namespace chwc
