#include "../../Converter.h"
#include "../Utils/Type.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/WithColor.h"

namespace chwc {

auto portFieldName(const clang::FieldDecl *owner, const clang::FieldDecl *leaf)
    -> std::string {
  std::string ownerName = owner->getNameAsString();
  std::string leafName = leaf->getNameAsString();

  return ownerName + "_" + leafName;
}

auto getIntegerInitValue(clang::Expr *expr) -> std::optional<int64_t> {
  if (!expr) {
    return std::nullopt;
  }

  expr = expr->IgnoreParenImpCasts();

  if (auto *intLit = llvm::dyn_cast<clang::IntegerLiteral>(expr)) {
    return intLit->getValue().getSExtValue();
  }

  if (auto *construct = llvm::dyn_cast<clang::CXXConstructExpr>(expr)) {
    if (construct->getNumArgs() == 1) {
      return getIntegerInitValue(construct->getArg(0));
    }
  }

  if (auto *temporary = llvm::dyn_cast<clang::MaterializeTemporaryExpr>(expr)) {
    return getIntegerInitValue(temporary->getSubExpr());
  }

  if (auto *bind = llvm::dyn_cast<clang::CXXBindTemporaryExpr>(expr)) {
    return getIntegerInitValue(bind->getSubExpr());
  }

  if (auto *cleanups = llvm::dyn_cast<clang::ExprWithCleanups>(expr)) {
    return getIntegerInitValue(cleanups->getSubExpr());
  }

  if (auto *initList = llvm::dyn_cast<clang::InitListExpr>(expr)) {
    if (initList->getNumInits() == 1) {
      return getIntegerInitValue(initList->getInit(0));
    }
  }

  return std::nullopt;
}

static void collectFieldInitializers(clang::FieldDecl *fieldDecl,
                                     HWFieldInfo &info) {
  if (!fieldDecl || !fieldDecl->hasInClassInitializer()) {
    return;
  }

  clang::Expr *init = fieldDecl->getInClassInitializer();
  if (!init) {
    return;
  }

  init = init->IgnoreParenImpCasts();

  if (auto *initList = llvm::dyn_cast<clang::InitListExpr>(init)) {
    for (unsigned i = 0; i < initList->getNumInits(); ++i) {
      std::optional<int64_t> value = getIntegerInitValue(initList->getInit(i));

      if (!value) {
        llvm::WithColor::error()
            << "chwc: register array initializer only supports integer "
               "literals: "
            << info.name << "\n";
        continue;
      }

      info.regInitValues.push_back(*value);
    }

    return;
  }

  if (std::optional<int64_t> value = getIntegerInitValue(init)) {
    info.regInitValue = *value;
    return;
  }

  llvm::WithColor::error()
      << "chwc: register initializer only supports integer literals: "
      << info.name << "\n";
}

auto fillHardwareFieldInfo(CHWConverter &converter,
                           HWModuleContext &moduleContext,
                           clang::FieldDecl *fieldDecl,
                           clang::FieldDecl *nameOwner, HWFieldInfo &info)
    -> bool {
  clang::QualType elementType = utils::getFieldElementType(fieldDecl);
  utils::SignalTypeInfo typeInfo = utils::getSignalTypeInfo(elementType);

  if (!typeInfo.isSignal || !typeInfo.fieldKind) {
    return false;
  }

  mlir::Type type = converter.convertType(fieldDecl->getType());
  if (!type) {
    return false;
  }

  info.fieldDecl = fieldDecl;
  info.name = nameOwner ? portFieldName(nameOwner, fieldDecl)
                        : fieldDecl->getNameAsString();
  info.kind = *typeInfo.fieldKind;
  info.type = type;
  info.elementType = type;
  info.isArray = false;
  info.arraySize = 1;
  info.regInitValue = 0;
  info.regInitValues.clear();

  if (std::optional<int64_t> init =
          utils::getRegInitValue(fieldDecl->getType())) {
    info.regInitValue = *init;
  }

  collectFieldInitializers(fieldDecl, info);

  if (std::optional<uint64_t> size = utils::getFieldArraySize(fieldDecl)) {
    info.isArray = true;
    info.arraySize = *size;

    auto arrayType = mlir::dyn_cast<circt::hw::ArrayType>(type);
    if (!arrayType) {
      llvm::WithColor::error()
          << "chwc: array hardware field did not lower to hw.array: "
          << info.name << "\n";
      return false;
    }

    info.elementType = arrayType.getElementType();
  }

  moduleContext.fields[fieldDecl] = info;
  moduleContext.fieldOrder.push_back(fieldDecl);
  return true;
}

auto CHWConverter::TraverseFieldDecl(clang::FieldDecl *fieldDecl) -> bool {
  if (!fieldDecl) {
    return true;
  }

  if (!moduleContext.recordDecl) {
    return true;
  }

  if (const clang::CXXRecordDecl *portDecl =
          utils::getPortRecordDecl(fieldDecl->getType())) {
    for (clang::FieldDecl *innerField :
         const_cast<clang::CXXRecordDecl *>(portDecl)->fields()) {
      HWFieldInfo info;

      if (!fillHardwareFieldInfo(*this, moduleContext, innerField, fieldDecl,
                                 info)) {
        llvm::WithColor::error()
            << "chwc: Port field must be Input, Output, Wire, Reg, "
               "or RegInit: "
            << innerField->getNameAsString() << "\n";
      }
    }

    return true;
  }

  if (const clang::CXXRecordDecl *submoduleDecl =
          utils::getInstanceModuleDecl(fieldDecl->getType())) {
    HWInstanceInfo instanceInfo;
    instanceInfo.fieldDecl = fieldDecl;
    instanceInfo.name = fieldDecl->getNameAsString();
    instanceInfo.moduleDecl = submoduleDecl;
    instanceInfo.moduleName = submoduleDecl->getNameAsString();

    for (clang::FieldDecl *portDecl :
         const_cast<clang::CXXRecordDecl *>(submoduleDecl)->fields()) {
      utils::SignalTypeInfo portTypeInfo =
          utils::getFieldElementTypeInfo(portDecl);

      if (!portTypeInfo.isSignal || !portTypeInfo.fieldKind) {
        continue;
      }

      if (*portTypeInfo.fieldKind == HWFieldKind::Input) {
        mlir::Type type = convertType(portDecl->getType());
        if (!type) {
          llvm::WithColor::error()
              << "chwc: failed to lower submodule input type: "
              << portDecl->getNameAsString() << "\n";
          continue;
        }

        instanceInfo.inputPorts.push_back(portDecl);
        instanceInfo.inputTypes.push_back(type);
        continue;
      }

      if (*portTypeInfo.fieldKind == HWFieldKind::Output) {
        mlir::Type type = convertType(portDecl->getType());
        if (!type) {
          llvm::WithColor::error()
              << "chwc: failed to lower submodule output type: "
              << portDecl->getNameAsString() << "\n";
          continue;
        }

        instanceInfo.outputPorts.push_back(portDecl);
        instanceInfo.outputTypes.push_back(type);
        continue;
      }
    }

    moduleContext.instanceOrder.push_back(fieldDecl);
    moduleContext.instances[fieldDecl] = std::move(instanceInfo);
    return true;
  }

  HWFieldInfo info;
  fillHardwareFieldInfo(*this, moduleContext, fieldDecl, nullptr, info);
  return true;
}

} // namespace chwc
