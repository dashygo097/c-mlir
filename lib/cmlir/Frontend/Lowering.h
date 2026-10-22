#ifndef CMLIR_FRONTEND_LOWERING_H
#define CMLIR_FRONTEND_LOWERING_H

#include "Calls/CallLowering.h"
#include "Pragmas/PragmaHandler.h"
#include "State/FunctionState.h"
#include "State/ModuleState.h"
#include "Types/TypeLowering.h"
#include "clang-mlir/Address.h"
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
        calls(context, types) {}

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
  CallLowering calls;
  bool trace = false;

  auto emitAddress(clang::Expr *expr) -> mlir::FailureOr<clang_mlir::Address>;
  auto emitArrayAddress(clang::ArraySubscriptExpr *expr)
      -> mlir::FailureOr<clang_mlir::Address>;
  auto emitAssignmentAddress(clang::BinaryOperator *assignOp)
      -> mlir::FailureOr<clang_mlir::Address>;
  auto emitMemberAddress(clang::MemberExpr *expr)
      -> mlir::FailureOr<clang_mlir::Address>;
  auto generateAddrOfUnaryOperator(clang::Expr *expr) -> mlir::Value;
  auto generateArraySubscriptExpr(clang::ArraySubscriptExpr *expr)
      -> mlir::Value;
  auto generateAssignmentBinaryOperator(clang::BinaryOperator *expr)
      -> mlir::Value;
  auto generateBinaryOperator(clang::BinaryOperator *binOp) -> mlir::Value;
  auto generateCastExpr(clang::CastExpr *expr) -> mlir::Value;
  auto generateCXXBoolLiteralExpr(clang::CXXBoolLiteralExpr *boolLit)
      -> mlir::Value;
  auto generateCXXConstructExpr(clang::CXXConstructExpr *constructExpr)
      -> mlir::Value;
  auto generateCXXThisExpr(clang::CXXThisExpr *thisExpr) -> mlir::Value;
  auto generateCallExpr(clang::CallExpr *callExpr) -> mlir::Value;
  auto generateCharacterLiteral(clang::CharacterLiteral *charLit)
      -> mlir::Value;
  auto generateConditionalOperator(clang::ConditionalOperator *condOp)
      -> mlir::Value;
  auto generateDeclRefExpr(clang::DeclRefExpr *declRef) -> mlir::Value;
  auto generateExpr(clang::Expr *expr) -> mlir::Value;
  auto generateFloatingLiteral(clang::FloatingLiteral *floatLit) -> mlir::Value;
  auto generateIncDecUnaryOperator(clang::UnaryOperator *expr) -> mlir::Value;
  auto generateInitListExpr(clang::InitListExpr *initList) -> mlir::Value;
  auto generateIntegerLiteral(clang::IntegerLiteral *intLit) -> mlir::Value;
  auto generateLogicalOperator(clang::BinaryOperator *expr) -> mlir::Value;
  auto generateMemberExpr(clang::MemberExpr *memberExpr) -> mlir::Value;
  auto generateParenExpr(clang::ParenExpr *parenExpr) -> mlir::Value;
  auto generatePureBinaryOperator(clang::BinaryOperator *binOp) -> mlir::Value;
  auto generateStringLiteral(clang::StringLiteral *strLit) -> mlir::Value;
  auto generateUnaryExprOrTypeTraitExpr(clang::UnaryExprOrTypeTraitExpr *expr)
      -> mlir::Value;
  auto generateUnaryOperator(clang::UnaryOperator *unOp) -> mlir::Value;
  auto loadLValue(clang::Expr *expr) -> mlir::Value;
  auto initializeObject(clang::QualType type, clang::Expr *initializer,
                        mlir::Value storage,
                        llvm::ArrayRef<mlir::Value> indices = {}) -> bool;
};

} // namespace cmlir::frontend

#endif // CMLIR_FRONTEND_LOWERING_H
