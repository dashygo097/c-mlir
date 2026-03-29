#ifndef CMLIRC_ASTVISITOR_H
#define CMLIRC_ASTVISITOR_H

#include "./Context/ContextManager.h"
#include "./Pragmas/PragmaHandler.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "clang/AST/RecursiveASTVisitor.h"

namespace cmlirc {

struct SimpleLoopInfo {
  const clang::VarDecl *inductionVar{nullptr};
  mlir::Value lowerBound;
  mlir::Value upperBound;
  mlir::Value step;
  bool isIncrementing{true};

  explicit operator bool() const {
    return inductionVar && lowerBound && upperBound && step;
  }
};

struct LoopContext {
  mlir::Block *headerBlock{nullptr};
  mlir::Block *exitBlock{nullptr};
  mlir::Value breakFlag{};
  mlir::Value continueFlag{};
  mlir::Value returnFlag{};
  mlir::Value returnValueSlot{};
};

struct SwitchArm {
  llvm::SmallVector<int64_t, 2> values;
  llvm::SmallVector<clang::Stmt *, 8> stmts;
  bool isDefault{false};
  bool hasBreak{false};
};

struct ArrayAccessInfo {
  mlir::Value base;
  llvm::SmallVector<mlir::Value, 4> indices{};
};

class CMLIRConverter : public clang::RecursiveASTVisitor<CMLIRConverter> {
public:
  explicit CMLIRConverter(ContextManager &contextManager,
                          LoopHintMap &loopHintMap)
      : contextManager(contextManager), loopHintMap(loopHintMap) {}
  ~CMLIRConverter() = default;

  // decl traits
  auto TraverseFunctionDecl(clang::FunctionDecl *funcDecl) -> bool;
  auto TraverseVarDecl(clang::VarDecl *varDecl) -> bool;
  auto TraverseRecordDecl(clang::RecordDecl *recordDecl) -> bool;

  // stmt traits
  auto TraverseStmt(clang::Stmt *stmt) -> bool;

  // return
  auto TraverseReturnStmt(clang::ReturnStmt *retStmt) -> bool;

  // control flow
  auto TraverseIfStmt(clang::IfStmt *ifStmt) -> bool;
  auto TraverseSwitchStmt(clang::SwitchStmt *switchStmt) -> bool;

  // loop
  auto TraverseForStmt(clang::ForStmt *forStmt) -> bool;
  auto TraverseWhileStmt(clang::WhileStmt *whileStmt) -> bool;
  auto TraverseDoStmt(clang::DoStmt *doStmt) -> bool;
  auto TraverseBreakStmt(clang::BreakStmt *breakStmt) -> bool;
  auto TraverseContinueStmt(clang::ContinueStmt *continueStmt) -> bool;

  // loop optimizations
  void emitLoopBodyWithIV(const clang::VarDecl *inductionVar,
                          mlir::Value ivIndex, mlir::Block *continueBlock,
                          clang::Stmt *body);
  void emitFullyUnrolledLoop(const SimpleLoopInfo &info, int64_t lb, int64_t ub,
                             int64_t st, clang::Stmt *body);
  void emitPartiallyUnrolledLoop(const SimpleLoopInfo &info, int64_t lb,
                                 int64_t ub, int64_t st, int64_t factor,
                                 clang::Stmt *body);
  void emitPlainForLoop(const SimpleLoopInfo &info, clang::Stmt *body);
  void emitWhileStyleForLoop(clang::ForStmt *forStmt);

private:
  ContextManager &contextManager;
  LoopHintMap &loopHintMap;

  // states
  llvm::DenseMap<const clang::VarDecl *, mlir::Value> symbolTable;
  llvm::DenseMap<const clang::ParmVarDecl *, mlir::Value> paramTable;
  llvm::DenseMap<const clang::FunctionDecl *, mlir::Value> functionTable;
  llvm::DenseMap<const clang::RecordDecl *, mlir::Type> recordTypeTable;
  llvm::DenseMap<const clang::RecordDecl *,
                 std::vector<const clang::FieldDecl *>>
      recordFieldTable;

  mlir::func::FuncOp currentFunc;
  mlir::Value *returnValueCapture;
  std::optional<ArrayAccessInfo> lastArrayAccess;

  llvm::SmallVector<LoopContext, 4> loopStack;

  // helpers
  auto getFieldIndex(const clang::RecordDecl *recordDecl,
                                        const clang::FieldDecl *fieldDecl) -> std::optional<uint32_t>;

  // type traits
  [[nodiscard]] auto convertType(clang::QualType type) -> mlir::Type;
  [[nodiscard]] auto convertBuiltinType(const clang::BuiltinType *type) -> mlir::Type;
  [[nodiscard]] auto convertArrayType(const clang::ArrayType *type) -> mlir::Type;
  [[nodiscard]] auto convertPointerType(const clang::PointerType *type) -> mlir::Type;
  [[nodiscard]] auto convertTypedefType(const clang::TypedefType *type) -> mlir::Type;
  [[nodiscard]] auto convertRecordType(const clang::RecordType *type) -> mlir::Type;

  // expr traits
  [[nodiscard]] auto hasSideEffects(clang::Expr *expr) const -> bool;
  auto generateExpr(clang::Expr *expr) -> mlir::Value;

  // paren
  auto generateParenExpr(clang::ParenExpr *parenExpr) -> mlir::Value;

  // literals
  auto generateCXXBoolLiteralExpr(clang::CXXBoolLiteralExpr *boolLit) -> mlir::Value;
  auto generateIntegerLiteral(clang::IntegerLiteral *intLit) -> mlir::Value;
  auto generateFloatingLiteral(clang::FloatingLiteral *floatLit) -> mlir::Value;
  auto generateCharacterLiteral(clang::CharacterLiteral *charLit) -> mlir::Value;
  auto generateStringLiteral(clang::StringLiteral *strLit) -> mlir::Value;

  // decl ref
  auto generateDeclRefExpr(clang::DeclRefExpr *declRef) -> mlir::Value;

  // implicit/cstyle cast
  auto generateImplicitCastExpr(clang::ImplicitCastExpr *castExpr) -> mlir::Value;
  auto generateCStyleCastExpr(clang::CStyleCastExpr *castExpr) -> mlir::Value;

  // array
  void storeInitListValues(clang::InitListExpr *initList, mlir::Value memref);
  auto generateInitListExpr(clang::InitListExpr *initList) -> mlir::Value;
  auto generateArraySubscriptExpr(clang::ArraySubscriptExpr *arraySub) -> mlir::Value;

  // unary
  auto generateUnaryOperator(clang::UnaryOperator *unOp) -> mlir::Value;
  auto generateAddrOfUnaryOperator(clang::Expr *addrOfOp) -> mlir::Value;
  auto generateIncDecUnaryOperator(clang::Expr *expr, bool isIncrement,
                                          bool isPrefix) -> mlir::Value;

  // binary
  auto generateBinaryOperator(clang::BinaryOperator *binOp) -> mlir::Value;
  auto generateAssignmentBinaryOperator(clang::BinaryOperator *assignOp) -> mlir::Value;
  auto generatePureBinaryOperator(clang::BinaryOperator *pureBinOp) -> mlir::Value;
  auto generateLAndBinaryOperator(mlir::Value lhs, mlir::Value rhs) -> mlir::Value;
  auto generateLOrBinaryOperator(mlir::Value lhs, mlir::Value rhs) -> mlir::Value;

  // conditional
  auto generateConditionalOperator(clang::ConditionalOperator *condOp) -> mlir::Value;

  // call
  auto generateCallExpr(clang::CallExpr *callExpr) -> mlir::Value;

  // struct
  auto generateCXXConstructExpr(clang::CXXConstructExpr *constructExpr) -> mlir::Value;
  auto generateMemberExpr(clang::MemberExpr *memberExpr) -> mlir::Value;

  // unary or type trait
  auto
  generateUnaryExprOrTypeTraitExpr(clang::UnaryExprOrTypeTraitExpr *traitExpr) -> mlir::Value;
};

} // namespace cmlirc

#endif // CMLIRC_ASTVISITOR_H
