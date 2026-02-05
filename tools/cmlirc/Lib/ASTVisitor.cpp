#include "./ASTVisitor.h"

namespace cmlirc {
using namespace clang;

CMLIRCASTVisitor::CMLIRCASTVisitor(clang::ASTContext *Context,
                                   MLIRContextManager &mlirContext)
    : clang_context_(Context), mlir_context_manager_(mlirContext),
      type_converter_(mlirContext.Builder()) {}

} // namespace cmlirc
