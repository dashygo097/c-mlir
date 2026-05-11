#include "../../Converter.h"
#include "../Utils/Type.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "llvm/Support/WithColor.h"

namespace chwc {

auto CHWConverter::TraverseFieldDecl(clang::FieldDecl *fieldDecl) -> bool {
  if (!fieldDecl) {
    return true;
  }

  if (!moduleContext.recordDecl) {
    return true;
  }

  if (const clang::CXXRecordDecl *bundleDecl =
          utils::getPortRecordDecl(fieldDecl->getType())) {
    for (clang::FieldDecl *innerField :
         const_cast<clang::CXXRecordDecl *>(bundleDecl)->fields()) {
      clang::QualType innerElementType = utils::getFieldElementType(innerField);

      utils::SignalTypeInfo typeInfo =
          utils::getSignalTypeInfo(innerElementType);

      if (!typeInfo.isSignal || !typeInfo.fieldKind) {
        llvm::WithColor::error()
            << "chwc: Port field must be Input, Output, Wire, Reg, "
               "or RegInit: "
            << innerField->getNameAsString() << "\n";
        continue;
      }

      mlir::Type innerType = convertType(innerField->getType());
      if (!innerType) {
        llvm::WithColor::error() << "chwc: failed to lower Port field type: "
                                 << innerField->getNameAsString() << "\n";
        continue;
      }

      HWFieldInfo info;
      info.fieldDecl = innerField;
      info.name =
          fieldDecl->getNameAsString() + "_" + innerField->getNameAsString();
      info.kind = *typeInfo.fieldKind;
      info.type = innerType;
      info.elementType = innerType;
      info.isArray = false;
      info.arraySize = 1;
      info.regInitValue = 0;

      if (std::optional<int64_t> init =
              utils::getRegInitValue(innerField->getType())) {
        info.regInitValue = *init;
      }

      if (std::optional<uint64_t> size = utils::getFieldArraySize(innerField)) {
        info.isArray = true;
        info.arraySize = *size;

        auto arrayType = mlir::dyn_cast<circt::hw::ArrayType>(innerType);
        if (!arrayType) {
          llvm::WithColor::error()
              << "chwc: Port array field did not lower to hw.array: "
              << info.name << "\n";
          continue;
        }

        info.elementType = arrayType.getElementType();
      }

      moduleContext.fields[innerField] = info;
      moduleContext.fieldOrder.push_back(innerField);
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

  mlir::Type type = convertType(fieldDecl->getType());
  if (!type) {
    return true;
  }

  clang::QualType elementType = utils::getFieldElementType(fieldDecl);
  utils::SignalTypeInfo typeInfo = utils::getSignalTypeInfo(elementType);

  if (!typeInfo.isSignal || !typeInfo.fieldKind) {
    return true;
  }

  HWFieldInfo info;
  info.fieldDecl = fieldDecl;
  info.name = fieldDecl->getNameAsString();
  info.kind = *typeInfo.fieldKind;
  info.type = type;
  info.elementType = type;
  info.isArray = false;
  info.arraySize = 1;
  info.regInitValue = 0;

  if (std::optional<int64_t> init =
          utils::getRegInitValue(fieldDecl->getType())) {
    info.regInitValue = *init;
  }

  if (std::optional<uint64_t> size = utils::getFieldArraySize(fieldDecl)) {
    info.isArray = true;
    info.arraySize = *size;

    auto arrayType = mlir::dyn_cast<circt::hw::ArrayType>(type);
    if (!arrayType) {
      llvm::WithColor::error()
          << "chwc: array hardware field did not lower to hw.array: "
          << info.name << "\n";
      return true;
    }

    info.elementType = arrayType.getElementType();
  }

  moduleContext.fields[fieldDecl] = info;
  moduleContext.fieldOrder.push_back(fieldDecl);
  return true;
}

} // namespace chwc
