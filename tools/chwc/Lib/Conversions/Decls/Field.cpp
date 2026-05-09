#include "../../Converter.h"
#include "../Utils/Type.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "llvm/Support/WithColor.h"

namespace chwc {

auto CHWConverter::TraverseFieldDecl(clang::FieldDecl *fieldDecl) -> bool {
  if (!fieldDecl || !moduleContext.recordDecl) {
    return true;
  }

  utils::SignalTypeInfo elemInfo = utils::getFieldElementTypeInfo(fieldDecl);

  std::optional<HWFieldKind> kind;

  if (elemInfo.isSignal && elemInfo.fieldKind) {
    kind = elemInfo.fieldKind;
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
