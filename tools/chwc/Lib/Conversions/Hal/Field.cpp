#include "../../Converter.h"
#include "../Utils/Annotation.h"
#include "../Utils/Expr.h"
#include "../Utils/Module.h"
#include "../Utils/State.h"

namespace chwc {

auto CHWConverter::classifyField(clang::FieldDecl *fieldDecl)
    -> std::optional<HWFieldKind> {
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
  expr = utils::ignoreCasts(expr);

  if (auto *memberExpr = mlir::dyn_cast_or_null<clang::MemberExpr>(expr)) {
    return mlir::dyn_cast<clang::FieldDecl>(memberExpr->getMemberDecl());
  }

  if (auto *declRef = mlir::dyn_cast_or_null<clang::DeclRefExpr>(expr)) {
    return mlir::dyn_cast<clang::FieldDecl>(declRef->getDecl());
  }

  return nullptr;
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
      mlir::Value value =
          utils::emitRegister(builder, loc, fieldInfo, fieldInfo.resetValue);

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

} // namespace chwc
