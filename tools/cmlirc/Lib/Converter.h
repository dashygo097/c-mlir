#ifndef CMLIRC_ASTVISITOR_H
#define CMLIRC_ASTVISITOR_H

#include "./Context/ContextManager.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "clang/AST/RecursiveASTVisitor.h"

namespace cmlirc {

class CMLIRConverter : public clang::RecursiveASTVisitor<CMLIRConverter> {
public:
  explicit CMLIRConverter(ContextManager &ctx) : context_manager_(ctx) {}
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
  // bool TraverseWhileStmt(clang::WhileStmt *whileStmt);
  // bool TraverseDoStmt(clang::DoStmt *doStmt);
  bool TraverseForStmt(clang::ForStmt *forStmt);
  // bool TraverseBreakStmt(clang::BreakStmt *breakStmt);
  // bool TraverseContinueStmt(clang::ContinueStmt *continueStmt);

private:
  ContextManager &context_manager_;

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

  struct ArrayAccessInfo {
    mlir::Value base;
    llvm::SmallVector<mlir::Value, 4> indices;
  };
  std::optional<ArrayAccessInfo> lastArrayAccess_;

  struct LoopContext {
    mlir::Block *headerBlock;
    mlir::Block *exitBlock;
  };
  llvm::SmallVector<LoopContext, 4> loopStack_;

  // side effect analysis
  [[nodiscard]] bool hasSideEffects(clang::Expr *expr) const;
  std::optional<uint32_t> getFieldIndex(const clang::RecordDecl *recordDecl,
                                        const clang::FieldDecl *fieldDecl);

  // type traits
  [[nodiscard]] mlir::Type convertType(clang::QualType type);
  [[nodiscard]] mlir::Type convertBuiltinType(const clang::BuiltinType *type);
  [[nodiscard]] mlir::Type convertArrayType(const clang::ArrayType *type);
  [[nodiscard]] mlir::Type convertPointerType(const clang::PointerType *type);
  [[nodiscard]] mlir::Type convertTypedefType(const clang::TypedefType *type);
  [[nodiscard]] mlir::Type convertRecordType(const clang::RecordType *type);

  [[nodiscard]] mlir::Value convertToBool(mlir::Value value);

  // expr traits
  mlir::Value generateExpr(clang::Expr *expr);

  // paren
  mlir::Value generateParenExpr(clang::ParenExpr *parenExpr);

  // literals
  mlir::Value generateCXXBoolLiteralExpr(clang::CXXBoolLiteralExpr *boolLit);
  mlir::Value generateIntegerLiteral(clang::IntegerLiteral *intLit);
  mlir::Value generateFloatingLiteral(clang::FloatingLiteral *floatLit);
  mlir::Value generateCharacterLiteral(clang::CharacterLiteral *charLit);

  // decl ref
  mlir::Value generateDeclRefExpr(clang::DeclRefExpr *declRef);

  // implicit cast
  mlir::Value generateImplicitCastExpr(clang::ImplicitCastExpr *castExpr);

  // array
  void storeInitListValues(clang::InitListExpr *initList, mlir::Value memref);
  mlir::Value generateInitListExpr(clang::InitListExpr *initList);
  mlir::Value generateArraySubscriptExpr(clang::ArraySubscriptExpr *expr);

  // unary
  mlir::Value generateUnaryOperator(clang::UnaryOperator *unOp);
  mlir::Value generateIncrementDecrement(clang::Expr *expr, bool isIncrement,
                                         bool isPrefix);

  // binary
  mlir::Value generateBinaryOperator(clang::BinaryOperator *binOp);

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
