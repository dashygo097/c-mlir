#ifndef CHWC_ASTVISITOR_H
#define CHWC_ASTVISITOR_H

#include "./Context/ContextManager.h"
#include "circt/Dialect/HW/HWOps.h"
#include "mlir/IR/Value.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include <string>

namespace chwc {

enum class HWFieldKind {
  Input,
  Output,
  Wire,
  Reg,
};

enum class HWEmitMode {
  Normal,
  Reset,
};

struct HWFieldInfo {
  const clang::FieldDecl *fieldDecl{nullptr};
  std::string name;
  HWFieldKind kind{HWFieldKind::Wire};

  mlir::Type type{};
  mlir::Type elementType{};

  bool isArray{false};
  uint64_t arraySize{1};

  mlir::Value resetValue{};

  int64_t regInitValue{0};
};

struct HWInstanceInfo {
  const clang::FieldDecl *fieldDecl{nullptr};
  std::string name;

  const clang::CXXRecordDecl *moduleDecl{nullptr};
  std::string moduleName;

  llvm::SmallVector<const clang::FieldDecl *, 8> inputPorts;
  llvm::SmallVector<const clang::FieldDecl *, 8> outputPorts;

  llvm::SmallVector<mlir::Type, 8> inputTypes;
  llvm::SmallVector<mlir::Type, 8> outputTypes;

  llvm::DenseMap<const clang::FieldDecl *, mlir::Value> inputValues;
  llvm::DenseMap<const clang::FieldDecl *, mlir::Value> outputValues;

  mlir::Operation *instanceOp{nullptr};
};

struct HWModuleContext {
  const clang::CXXRecordDecl *recordDecl{nullptr};
  circt::hw::HWModuleOp moduleOp{};

  mlir::Value clock{};
  mlir::Value reset{};

  uint64_t anonymousRegIndex{0};

  llvm::DenseMap<const clang::FieldDecl *, HWFieldInfo> fields;
  llvm::SmallVector<const clang::FieldDecl *, 16> fieldOrder;

  llvm::DenseMap<const clang::FieldDecl *, mlir::Value> currentValues;
  llvm::DenseMap<const clang::FieldDecl *, mlir::Value> nextValues;
  llvm::DenseMap<const clang::FieldDecl *, mlir::Value> outputValues;

  llvm::DenseMap<const clang::FieldDecl *, HWInstanceInfo> instances;
  llvm::SmallVector<const clang::FieldDecl *, 8> instanceOrder;

  llvm::SmallVector<const clang::CXXMethodDecl *, 4> resetMethods;
  llvm::SmallVector<const clang::CXXMethodDecl *, 4> clockMethods;

  llvm::SmallVector<mlir::Attribute, 4> parameters;
  llvm::StringMap<mlir::TypedAttr> parameterRefs;

  void clear() {
    recordDecl = nullptr;
    moduleOp = nullptr;
    clock = nullptr;
    reset = nullptr;
    fields.clear();
    fieldOrder.clear();
    currentValues.clear();
    nextValues.clear();
    outputValues.clear();
    instances.clear();
    instanceOrder.clear();
    resetMethods.clear();
    clockMethods.clear();
    parameters.clear();
    parameterRefs.clear();
  }
};

struct HWFunctionContext {
  llvm::DenseMap<const clang::VarDecl *, mlir::Value> locals;
  mlir::Value returnValue{};
  bool hasReturnValue{false};
};

class CHWConverter : public clang::RecursiveASTVisitor<CHWConverter> {
public:
  explicit CHWConverter(CHWContextManager &contextManager)
      : contextManager(contextManager) {}
  ~CHWConverter() = default;

  // decl traits
  auto TraverseFunctionDecl(clang::FunctionDecl *functionDecl) -> bool;
  auto TraverseCXXRecordDecl(clang::CXXRecordDecl *recordDecl) -> bool;
  auto TraverseFieldDecl(clang::FieldDecl *fieldDecl) -> bool;

  // stmt traits
  auto TraverseStmt(clang::Stmt *stmt) -> bool;
  auto TraverseCompoundStmt(clang::CompoundStmt *compoundStmt) -> bool;
  auto TraverseDeclStmt(clang::DeclStmt *declStmt) -> bool;
  auto TraverseReturnStmt(clang::ReturnStmt *returnStmt) -> bool;

  // control flow
  auto TraverseIfStmt(clang::IfStmt *ifStmt) -> bool;

  // loop
  auto TraverseForStmt(clang::ForStmt *forStmt) -> bool;

private:
  CHWContextManager &contextManager;

  HWModuleContext moduleContext;
  llvm::SmallVector<HWFunctionContext, 8> functionStack;
  HWEmitMode emitMode{HWEmitMode::Normal};

  // type router
  auto convertType(clang::QualType type) -> mlir::Type;
  auto convertBuiltinType(const clang::BuiltinType *type) -> mlir::Type;

  // expr router
  auto generateExpr(clang::Expr *expr) -> mlir::Value;

  // expr dealers
  auto generateArraySubscriptExpr(clang::ArraySubscriptExpr *arraySub)
      -> mlir::Value;
  auto generateBinaryOperator(clang::BinaryOperator *binOp) -> mlir::Value;
  auto generateAssignmentBinaryOperator(clang::BinaryOperator *assignOp)
      -> mlir::Value;
  auto generatePureBinaryOperator(clang::BinaryOperator *binOp) -> mlir::Value;
  auto generateCXXBindTemporaryExpr(clang::CXXBindTemporaryExpr *expr)
      -> mlir::Value;
  auto generateCXXBoolLiteralExpr(clang::CXXBoolLiteralExpr *boolLit)
      -> mlir::Value;
  auto generateCXXConstructExpr(clang::CXXConstructExpr *constructExpr)
      -> mlir::Value;
  auto generateCXXFunctionalCastExpr(clang::CXXFunctionalCastExpr *castExpr)
      -> mlir::Value;
  auto generateCXXMemberCallExpr(clang::CXXMemberCallExpr *callExpr)
      -> mlir::Value;
  auto generateCXXOperatorCallExpr(clang::CXXOperatorCallExpr *callExpr)
      -> mlir::Value;
  auto generateCallExpr(clang::CallExpr *callExpr) -> mlir::Value;
  auto generateDeclRefExpr(clang::DeclRefExpr *declRef) -> mlir::Value;
  auto generateExprWithCleanups(clang::ExprWithCleanups *expr) -> mlir::Value;
  auto generateImplicitCastExpr(clang::ImplicitCastExpr *castExpr)
      -> mlir::Value;
  auto generateIntegerLiteral(clang::IntegerLiteral *intLit) -> mlir::Value;
  auto generateMaterializeTemporaryExpr(clang::MaterializeTemporaryExpr *expr)
      -> mlir::Value;
  auto generateMemberExpr(clang::MemberExpr *memberExpr) -> mlir::Value;
  auto generateUnaryOperator(clang::UnaryOperator *unOp) -> mlir::Value;

  auto generateLAndBinaryOperator(mlir::Value lhs, mlir::Value rhs)
      -> mlir::Value;
  auto generateLOrBinaryOperator(mlir::Value lhs, mlir::Value rhs)
      -> mlir::Value;
};

} // namespace chwc

#endif // CHWC_ASTVISITOR_H
