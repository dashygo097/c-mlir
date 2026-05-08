#include "../../Converter.h"

namespace chwc {

auto CHWConverter::TraverseFunctionDecl(clang::FunctionDecl *functionDecl)
    -> bool {
  (void)functionDecl;
  return true;
}

} // namespace chwc
