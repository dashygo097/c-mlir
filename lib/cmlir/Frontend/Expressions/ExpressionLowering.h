#ifndef CMLIR_FRONTEND_EXPRESSION_LOWERING_H
#define CMLIR_FRONTEND_EXPRESSION_LOWERING_H

#include "../Calls/CallLowering.h"
#include "../State/FunctionState.h"
#include "../Types/TypeLowering.h"
#include "clang-mlir/Address.h"
#include "clang/AST/ExprCXX.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include <variant>

namespace cmlir::frontend {

class ExpressionLowering {
public:
  ExpressionLowering(clang_mlir::Context &context, TypeLowering &types,
                     FunctionState &function, const bool &trace)
      : context(context), types(types), function(function), trace(trace),
        calls(context, types) {}

  auto emitValue(clang::Expr &expr) -> mlir::FailureOr<mlir::Value>;
  auto emitAddress(clang::Expr &expr) -> mlir::FailureOr<clang_mlir::Address>;
  auto emitDiscarded(clang::Expr &expr) -> mlir::LogicalResult;

private:
  clang_mlir::Context &context;
  TypeLowering &types;
  FunctionState &function;
  const bool &trace;
  CallLowering calls;

  struct ObjectResult {
    clang_mlir::Address address;
    mlir::Value computed;
  };

  using LValueResult = std::variant<mlir::Value, ObjectResult>;
  using LValueConsumer =
      llvm::function_ref<mlir::FailureOr<mlir::Value>(const LValueResult &)>;

  auto emitLValue(clang::Expr *expr, mlir::Type resultType,
                  LValueConsumer consume) -> mlir::FailureOr<mlir::Value>;
  auto emitConditionalLValue(clang::ConditionalOperator &expr,
                             mlir::Type resultType, LValueConsumer consume)
      -> mlir::FailureOr<mlir::Value>;
  auto emitAssignmentLValue(clang::BinaryOperator &expr, mlir::Type resultType,
                            LValueConsumer consume)
      -> mlir::FailureOr<mlir::Value>;
  auto emitIncDecLValue(clang::UnaryOperator &expr, mlir::Type resultType,
                        LValueConsumer consume) -> mlir::FailureOr<mlir::Value>;
  auto emitStorageAddress(clang::Expr *expr)
      -> mlir::FailureOr<clang_mlir::Address>;
  auto addressOf(clang::Expr *expr) -> mlir::FailureOr<clang_mlir::Address>;
  auto emitArrayAddress(clang::ArraySubscriptExpr *expr)
      -> mlir::FailureOr<clang_mlir::Address>;
  auto storeAssignment(clang::BinaryOperator *expr,
                       const clang_mlir::Address &address, mlir::Value rhs)
      -> mlir::Value;
  auto updateIncDec(clang::UnaryOperator *expr,
                    const clang_mlir::Address &address) -> mlir::Value;
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
  auto generateCallExpr(clang::CallExpr *callExpr)
      -> mlir::FailureOr<mlir::Value>;
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
};

} // namespace cmlir::frontend

#endif // CMLIR_FRONTEND_EXPRESSION_LOWERING_H
