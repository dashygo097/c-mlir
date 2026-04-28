#include "../../Converter.h"
#include "../Utils/HWOps.h"
#include "clang/AST/Attr.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/WithColor.h"

namespace chwc {

auto CHWConverter::TraverseCXXRecordDecl(clang::CXXRecordDecl *recordDecl)
    -> bool {
  if (!recordDecl || !recordDecl->isCompleteDefinition()) {
    return true;
  }

  if (!isHardwareClass(recordDecl)) {
    return true;
  }

  currentRecordDecl = recordDecl;

  collectHardwareClass(recordDecl);
  emitHardwareClass(recordDecl);

  currentRecordDecl = nullptr;
  return true;
}

} // namespace chwc
