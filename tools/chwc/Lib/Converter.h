#ifndef CHWC_ASTVISITOR_H
#define CHWC_ASTVISITOR_H

#include "./Context/ContextManager.h"
#include "mlir/IR/Value.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "llvm/ADT/DenseMap.h"
#include <string>

namespace chwc {

enum class HWFieldKind {
  Input,
  Output,
  Reg,
  Wire,
};

struct HWFieldInfo {
  const clang::FieldDecl *fieldDecl{nullptr};
  std::string name;
  mlir::Type type;
  HWFieldKind kind{HWFieldKind::Wire};
  mlir::Value resetValue{};
};

class CHWConverter : public clang::RecursiveASTVisitor<CHWConverter> {
public:
  explicit CHWConverter(CHWContextManager &contextManager)
      : contextManager(contextManager) {}
  ~CHWConverter() = default;

  // decl traits
  auto TraverseFunctionDecl(clang::FunctionDecl *functionDecl) -> bool;
  auto TraverseCXXRecordDecl(clang::CXXRecordDecl *recordDecl) -> bool;

  // stmt traits
  auto TraverseStmt(clang::Stmt *stmt) -> bool;
  auto TraverseCompoundStmt(clang::CompoundStmt *compoundStmt) -> bool;
  auto TraverseDeclStmt(clang::DeclStmt *declStmt) -> bool;

  // control flow
  auto TraverseIfStmt(clang::IfStmt *ifStmt) -> bool;

  // loop

  // loop optimizations

private:
  CHWContextManager &contextManager;

  // states
  const clang::CXXRecordDecl *currentRecordDecl{nullptr};
  const clang::CXXMethodDecl *resetMethod{nullptr};
  const clang::CXXMethodDecl *clockTickMethod{nullptr};

  llvm::DenseMap<const clang::FieldDecl *, HWFieldInfo> fieldTable;
  llvm::DenseMap<const clang::FieldDecl *, mlir::Value> currentFieldValueTable;
  llvm::DenseMap<const clang::FieldDecl *, mlir::Value> nextFieldValueTable;
  llvm::DenseMap<const clang::FieldDecl *, mlir::Value> outputValueTable;
  llvm::DenseMap<const clang::VarDecl *, mlir::Value> localValueTable;

  // Hareware abstraction layer
  // module traits
  auto isHardwareClass(clang::CXXRecordDecl *recordDecl) -> bool;
  void collectHardwareClass(clang::CXXRecordDecl *recordDecl);
  void emitHardwareClass(clang::CXXRecordDecl *recordDecl);

  // clock traits
  void emitClockTick();

  // reset traits
  void collectResetValues();

  // field traits
  void emitStateDecls();
  auto getAssignedField(clang::Expr *expr) -> const clang::FieldDecl *;

  // type traits
  auto convertType(clang::QualType type) -> mlir::Type;
  auto convertBuiltinType(const clang::BuiltinType *type) -> mlir::Type;

  // expr traits
  auto generateExpr(clang::Expr *expr) -> mlir::Value;

  auto generateCXXBoolLiteralExpr(clang::CXXBoolLiteralExpr *boolLit)
      -> mlir::Value;
  auto generateIntegerLiteral(clang::IntegerLiteral *intLit) -> mlir::Value;

  auto generateDeclRefExpr(clang::DeclRefExpr *declRef) -> mlir::Value;
  auto generateImplicitCastExpr(clang::ImplicitCastExpr *castExpr)
      -> mlir::Value;
  auto generateMemberExpr(clang::MemberExpr *memberExpr) -> mlir::Value;

  auto generateBinaryOperator(clang::BinaryOperator *binOp) -> mlir::Value;
  auto generateAssignmentBinaryOperator(clang::BinaryOperator *assignOp)
      -> mlir::Value;
};

} // namespace chwc

#endif // CHWC_ASTVISITOR_H
