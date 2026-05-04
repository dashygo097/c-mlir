#ifndef CHWC_ASTVISITOR_H
#define CHWC_ASTVISITOR_H

#include "./Context/ContextManager.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Support/BackedgeBuilder.h"
#include "mlir/IR/Value.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include <memory>
#include <optional>
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
  auto TraverseReturnStmt(clang::ReturnStmt *returnStmt) -> bool;

  // control flow
  auto TraverseIfStmt(clang::IfStmt *ifStmt) -> bool;

  // loop

  // loop optimizations

private:
  CHWContextManager &contextManager;

  // states
  const clang::CXXRecordDecl *currentRecordDecl{nullptr};

  llvm::SmallVector<clang::CXXMethodDecl *, 4> resetMethods;
  llvm::SmallVector<clang::CXXMethodDecl *, 4> clockTickMethods;

  circt::hw::HWModuleOp currentModuleOp;
  mlir::Value clockValue;
  mlir::Value resetValue;

  std::unique_ptr<circt::BackedgeBuilder> backedgeBuilder;
  llvm::DenseMap<const clang::FieldDecl *, circt::Backedge>
      registerNextBackedgeTable;

  llvm::DenseMap<const clang::FieldDecl *, mlir::Value> inputValueTable;
  llvm::SmallVector<mlir::Value, 8> outputValues;

  llvm::DenseMap<const clang::FieldDecl *, HWFieldInfo> fieldTable;
  llvm::SmallVector<const clang::FieldDecl *, 8> hardwareFieldOrder;
  llvm::DenseMap<const clang::FieldDecl *, mlir::Value> currentFieldValueTable;
  llvm::DenseMap<const clang::FieldDecl *, mlir::Value> nextFieldValueTable;
  llvm::DenseMap<const clang::FieldDecl *, mlir::Value> outputValueTable;
  llvm::DenseMap<const clang::VarDecl *, mlir::Value> localValueTable;

  mlir::Value currentReturnValue{};
  bool hasCurrentReturnValue{false};
  unsigned helperInlineDepth{0};

  void clearHardwareState() {
    currentModuleOp = nullptr;
    clockValue = nullptr;
    resetValue = nullptr;

    if (backedgeBuilder) {
      backedgeBuilder->abandon();
    }
    backedgeBuilder.reset();

    registerNextBackedgeTable.clear();
    inputValueTable.clear();
    outputValues.clear();

    fieldTable.clear();
    hardwareFieldOrder.clear();
    currentFieldValueTable.clear();
    nextFieldValueTable.clear();
    outputValueTable.clear();
    localValueTable.clear();

    currentReturnValue = nullptr;
    hasCurrentReturnValue = false;
    helperInlineDepth = 0;

    resetMethods.clear();
    clockTickMethods.clear();
  }

  // Hardware abstraction layer
  // module traits
  auto isHardwareClass(clang::CXXRecordDecl *recordDecl) -> bool;
  void collectHardwareClass(clang::CXXRecordDecl *recordDecl);
  void emitHardwareClass(clang::CXXRecordDecl *recordDecl);

  // clock traits
  void emitClockTick();

  // reset traits
  void collectResetValues();

  // field traits
  auto classifyField(clang::FieldDecl *fieldDecl) -> std::optional<HWFieldKind>;
  auto getAssignedField(clang::Expr *expr) -> const clang::FieldDecl *;
  void emitStateDecls();

  // type traits
  auto convertType(clang::QualType type) -> mlir::Type;
  auto convertBuiltinType(const clang::BuiltinType *type) -> mlir::Type;

  // expr traits
  auto generateExpr(clang::Expr *expr) -> mlir::Value;

  auto generateExprWithCleanups(clang::ExprWithCleanups *expr) -> mlir::Value;
  auto generateCXXBindTemporaryExpr(clang::CXXBindTemporaryExpr *expr)
      -> mlir::Value;
  auto generateMaterializeTemporaryExpr(clang::MaterializeTemporaryExpr *expr)
      -> mlir::Value;
  auto generateCXXConstructExpr(clang::CXXConstructExpr *expr) -> mlir::Value;

  auto generateCXXBoolLiteralExpr(clang::CXXBoolLiteralExpr *boolLit)
      -> mlir::Value;
  auto generateIntegerLiteral(clang::IntegerLiteral *intLit) -> mlir::Value;

  auto generateDeclRefExpr(clang::DeclRefExpr *declRef) -> mlir::Value;
  auto generateImplicitCastExpr(clang::ImplicitCastExpr *castExpr)
      -> mlir::Value;
  auto generateMemberExpr(clang::MemberExpr *memberExpr) -> mlir::Value;
  auto generateCXXMemberCallExpr(clang::CXXMemberCallExpr *callExpr)
      -> mlir::Value;
  auto generateCXXOperatorCallExpr(clang::CXXOperatorCallExpr *callExpr)
      -> mlir::Value;

  auto generateUnaryOperator(clang::UnaryOperator *unOp) -> mlir::Value;
  auto generateIncDecUnaryOperator(clang::Expr *expr, bool isIncrement,
                                   bool isPrefix) -> mlir::Value;
  auto generateAddrOfUnaryOperator(clang::Expr *addrOfExpr) -> mlir::Value;

  auto generateBinaryOperator(clang::BinaryOperator *binOp) -> mlir::Value;
  auto generateAssignmentBinaryOperator(clang::BinaryOperator *assignOp)
      -> mlir::Value;
  auto generatePureBinaryOperator(clang::BinaryOperator *binOp) -> mlir::Value;
  auto generateCompoundAssignmentBinaryOperator(
      clang::CompoundAssignOperator *compoundOp) -> mlir::Value;

  auto generateLAndBinaryOperator(mlir::Value lhs, mlir::Value rhs)
      -> mlir::Value;
  auto generateLOrBinaryOperator(mlir::Value lhs, mlir::Value rhs)
      -> mlir::Value;
};

} // namespace chwc

#endif // CHWC_ASTVISITOR_H
