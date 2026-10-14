#ifndef CLANG_SUPPORT_DISCOVERY_H
#define CLANG_SUPPORT_DISCOVERY_H

#include "llvm/ADT/SmallVector.h"

namespace clang {

class ASTContext;
class FunctionDecl;

} // namespace clang

namespace clang_support {

// Discovery may walk recursively. Emission is deliberately a separate concern.
auto definedFunctions(clang::ASTContext &ast)
    -> llvm::SmallVector<clang::FunctionDecl *>;
auto directDefinedCallees(clang::FunctionDecl &function)
    -> llvm::SmallVector<clang::FunctionDecl *>;

} // namespace clang_support

#endif // CLANG_SUPPORT_DISCOVERY_H
