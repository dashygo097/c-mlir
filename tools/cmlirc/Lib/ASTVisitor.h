#ifndef CMLIRC_ASTVISITOR_H
#define CMLIRC_ASTVISITOR_H

#include "./MLIRContextManager.h"
#include "./TypeConverter.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "clang/AST/RecursiveASTVisitor.h"

namespace cmlirc {

class CMLIRCASTVisitor : public clang::RecursiveASTVisitor<CMLIRCASTVisitor> {
public:
  explicit CMLIRCASTVisitor(clang::ASTContext *Context,
                            MLIRContextManager &mlirCtx);
  ~CMLIRCASTVisitor() = default;

  clang::ASTContext *clang_context_;
  MLIRContextManager &mlir_context_manager_;
  TypeConverter type_converter_;

private:
  // states
  llvm::DenseMap<const clang::VarDecl *, mlir::Value> symbolTable;
  mlir::func::FuncOp currentFunc;
};

} // namespace cmlirc

#endif // CMLIRC_ASTVISITOR_H
