#include "../../Converter.h"
#include "../Utils/Annotation.h"
#include "../Utils/Type.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "llvm/Support/WithColor.h"

namespace chwc {

auto CHWConverter::TraverseFieldDecl(clang::FieldDecl *fieldDecl) -> bool {
  if (!fieldDecl || !moduleContext.recordDecl) {
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

  utils::SignalTypeInfo elemInfo = utils::getFieldElementTypeInfo(fieldDecl);

  std::optional<HWFieldKind> kind;

  if (elemInfo.isSignal && elemInfo.fieldKind) {
    kind = elemInfo.fieldKind;
  } else if (std::optional<std::string> anno =
                 utils::getAnnotation(fieldDecl)) {
    if (*anno == "hw.input") {
      kind = HWFieldKind::Input;
    } else if (*anno == "hw.output") {
      kind = HWFieldKind::Output;
    } else if (*anno == "hw.wire") {
      kind = HWFieldKind::Wire;
    } else if (*anno == "hw.reg") {
      kind = HWFieldKind::Reg;
    }
  }

  if (!kind) {
    return true;
  }

  mlir::Type type = convertType(fieldDecl->getType());
  if (!type) {
    llvm::WithColor::error() << "chwc: unsupported hardware field type: "
                             << fieldDecl->getType().getAsString() << "\n";
    return true;
  }

  HWFieldInfo info;
  info.fieldDecl = fieldDecl;
  info.name = fieldDecl->getNameAsString();
  info.kind = *kind;
  info.type = type;

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
          << "chwc: array hardware field did not lower to hw.array\n";
      return true;
    }

    info.elementType = arrayType.getElementType();
  } else {
    info.elementType = type;
  }

  moduleContext.fieldOrder.push_back(fieldDecl);
  moduleContext.fields[fieldDecl] = info;
  return true;
}

} // namespace chwc
