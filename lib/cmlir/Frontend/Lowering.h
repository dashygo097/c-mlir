#ifndef CMLIR_FRONTEND_LOWERING_H
#define CMLIR_FRONTEND_LOWERING_H

#include "Expressions/ExpressionLowering.h"
#include "Pragmas/PragmaHandler.h"
#include "State/FunctionState.h"
#include "State/ModuleState.h"
#include "Types/TypeLowering.h"
#include "clang-mlir/Context.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/Stmt.h"

namespace cmlir::frontend {

class ASTLowering {
public:
  explicit ASTLowering(clang_mlir::Context &context,
                       const LoopHintMap &loopHintMap)
      : context(context), loopHintMap(loopHintMap), types(context),
        expressions(context, types, function, trace) {}

  ~ASTLowering() = default;

  auto emitDeclaration(clang::Decl *decl) -> bool;
  auto emitFunction(clang::FunctionDecl *decl) -> bool;
  auto emitVariable(clang::VarDecl *decl) -> bool;
  auto emitRecord(clang::RecordDecl *decl) -> bool;
  auto emitEnum(clang::EnumDecl *decl) -> bool;
  auto emitStatement(clang::Stmt *stmt) -> bool;
  auto emitReturn(clang::ReturnStmt *stmt) -> bool;
  auto emitIf(clang::IfStmt *stmt) -> bool;
  auto emitSwitch(clang::SwitchStmt *stmt) -> bool;
  auto emitFor(clang::ForStmt *stmt) -> bool;
  auto emitWhile(clang::WhileStmt *stmt) -> bool;
  auto emitDo(clang::DoStmt *stmt) -> bool;
  auto emitBreak(clang::BreakStmt *stmt) -> bool;
  auto emitContinue(clang::ContinueStmt *stmt) -> bool;
  auto emitCFGLoop(clang::Expr *condition, clang::Stmt *body,
                   clang::Expr *increment, bool bodyFirst) -> bool;

  auto setTrace(bool value) -> void { trace = value; }

  [[nodiscard]] auto failed() const -> bool {
    return context.source().failed();
  }

private:
  clang_mlir::Context &context;
  const LoopHintMap &loopHintMap;
  ModuleState module;
  FunctionState function;
  TypeLowering types;
  ExpressionLowering expressions;
  bool trace = false;
};

} // namespace cmlir::frontend

#endif // CMLIR_FRONTEND_LOWERING_H
