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
};

struct ArrayAccessInfo {
  mlir::Value base;
  llvm::SmallVector<mlir::Value, 4> indices{};
};

enum class BreakTargetKind { ScfYield, CfBranch };
struct BreakTarget {
  BreakTargetKind kind;
  mlir::Block *block{nullptr};
};

class CMLIRConverter : public clang::RecursiveASTVisitor<CMLIRConverter> {
public:
  explicit CMLIRConverter(ContextManager &ctx, LoopHintMap &loopHints)
      : context_manager_(ctx), loop_hints_(loopHints) {}
  ~CMLIRConverter() = default;

  // decl traits
  bool TraverseFunctionDecl(clang::FunctionDecl *D);
  bool TraverseVarDecl(clang::VarDecl *decl);
  bool TraverseRecordDecl(clang::RecordDecl *recordDecl);

  // stmt traits
  bool TraverseStmt(clang::Stmt *stmt);

  // return
  bool TraverseReturnStmt(clang::ReturnStmt *retStmt);

  // control flow
  bool TraverseIfStmt(clang::IfStmt *ifStmt);
  bool TraverseSwitchStmt(clang::SwitchStmt *switchStmt);
  bool TraverseWhileStmt(clang::WhileStmt *whileStmt);
  bool TraverseDoStmt(clang::DoStmt *doStmt);
  bool TraverseForStmt(clang::ForStmt *forStmt);
  bool TraverseBreakStmt(clang::BreakStmt *breakStmt);
  // bool TraverseContinueStmt(clang::ContinueStmt *continueStmt);

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
  ContextManager &context_manager_;
  LoopHintMap &loop_hints_;

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

  llvm::SmallVector<BreakTarget, 4> breakStack;

  // helpers
  std::optional<uint32_t> getFieldIndex(const clang::RecordDecl *recordDecl,
                                        const clang::FieldDecl *fieldDecl);

  // type traits
  [[nodiscard]] mlir::Type convertType(const clang::QualType type);
  [[nodiscard]] mlir::Type convertBuiltinType(const clang::BuiltinType *type);
  [[nodiscard]] mlir::Type convertArrayType(const clang::ArrayType *type);
  [[nodiscard]] mlir::Type convertPointerType(const clang::PointerType *type);
  [[nodiscard]] mlir::Type convertTypedefType(const clang::TypedefType *type);
  [[nodiscard]] mlir::Type convertRecordType(const clang::RecordType *type);

  // expr traits
  [[nodiscard]] bool hasSideEffects(clang::Expr *expr) const;
  mlir::Value generateExpr(clang::Expr *expr);

  // paren
  mlir::Value generateParenExpr(clang::ParenExpr *parenExpr);

  // literals
  mlir::Value generateCXXBoolLiteralExpr(clang::CXXBoolLiteralExpr *boolLit);
  mlir::Value generateIntegerLiteral(clang::IntegerLiteral *intLit);
  mlir::Value generateFloatingLiteral(clang::FloatingLiteral *floatLit);
  mlir::Value generateCharacterLiteral(clang::CharacterLiteral *charLit);
  mlir::Value generateStringLiteral(clang::StringLiteral *strLit);

  // decl ref
  mlir::Value generateDeclRefExpr(clang::DeclRefExpr *declRef);

  // implicit/cstyle cast
  mlir::Value generateImplicitCastExpr(clang::ImplicitCastExpr *castExpr);
  mlir::Value generateCStyleCastExpr(clang::CStyleCastExpr *castExpr);

  // array
  void storeInitListValues(clang::InitListExpr *initList, mlir::Value memref);
  mlir::Value generateInitListExpr(clang::InitListExpr *initList);
  mlir::Value generateArraySubscriptExpr(clang::ArraySubscriptExpr *expr);

  // unary
  mlir::Value generateUnaryOperator(clang::UnaryOperator *unOp);
  mlir::Value generateAddrOfUnaryOperator(clang::Expr *addrOfOp);
  mlir::Value generateIncDecUnaryOperator(clang::Expr *expr, bool isIncrement,
                                          bool isPrefix);

  // binary
  mlir::Value generateBinaryOperator(clang::BinaryOperator *binOp);
  mlir::Value generateAssignmentBinaryOperator(clang::BinaryOperator *assignOp);
  mlir::Value generatePureBinaryOperator(clang::BinaryOperator *pureBinOp);
  mlir::Value generateLAndBinaryOperator(mlir::Value lhs, mlir::Value rhs);
  mlir::Value generateLOrBinaryOperator(mlir::Value lhs, mlir::Value rhs);

  // conditional
  mlir::Value generateConditionalOperator(clang::ConditionalOperator *condOp);

  // call
  mlir::Value generateCallExpr(clang::CallExpr *callExpr);

  // struct
  mlir::Value generateCXXConstructExpr(clang::CXXConstructExpr *constructExpr);
  mlir::Value generateMemberExpr(clang::MemberExpr *memberExpr);
};

} // namespace cmlirc

#endif // CMLIRC_ASTVISITOR_H
