#include "../../Converter.h"

namespace chwc {

auto CHWConverter::TraverseFunctionDecl(clang::FunctionDecl *functionDecl)
    -> bool {
  return true;
}

} // namespace chwc
