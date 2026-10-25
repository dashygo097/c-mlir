#include "ExpressionLowering.h"
#include "clang-mlir/Casts.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

namespace cmlir::frontend {

auto ExpressionLowering::emitConditionalLValue(clang::ConditionalOperator &expr,
                                               mlir::Type resultType,
                                               LValueConsumer consume)
    -> mlir::FailureOr<mlir::Value> {
  auto &builder = context.builder();
  auto loc = context.source().location(expr.getExprLoc());
  auto condition =
      clang_mlir::toBool(builder, loc, generateExpr(expr.getCond()));
  if (!condition || !resultType) {
    return mlir::failure();
  }
  bool hasResult = !mlir::isa<mlir::NoneType>(resultType);
  auto selection = mlir::scf::IfOp::create(
      builder, loc, hasResult ? mlir::TypeRange{resultType} : mlir::TypeRange{},
      condition, true);
  for (auto [block, arm] :
       {std::pair{selection.thenBlock(), expr.getTrueExpr()},
        std::pair{selection.elseBlock(), expr.getFalseExpr()}}) {
    mlir::OpBuilder::InsertionGuard guard(builder);
    if (!block->empty() && llvm::isa<mlir::scf::YieldOp>(block->back())) {
      block->back().erase();
    }
    builder.setInsertionPointToStart(block);
    auto emitted = emitLValue(arm, resultType, consume);
    if (mlir::failed(emitted)) {
      return mlir::failure();
    }
    mlir::Value value;
    if (hasResult) {
      value = clang_mlir::toValue(builder, loc, *emitted, resultType);
      if (!value) {
        return mlir::failure();
      }
    }
    mlir::scf::YieldOp::create(
        builder, loc, hasResult ? mlir::ValueRange{value} : mlir::ValueRange{});
  }
  builder.setInsertionPointAfter(selection);
  return hasResult ? selection.getResult(0) : mlir::Value{};
}

auto ExpressionLowering::generateConditionalOperator(
    clang::ConditionalOperator *expr) -> mlir::Value {
  if (expr->isGLValue()) {
    if (expr->getType()->isArrayType()) {
      auto address = addressOf(expr);
      return mlir::succeeded(address)
                 ? address->materialize(
                       context.builder(),
                       context.source().location(expr->getExprLoc()))
                 : mlir::Value{};
    }
    return loadLValue(expr);
  }
  auto &builder = context.builder();
  auto loc = context.source().location(expr->getExprLoc());
  auto condition =
      clang_mlir::toBool(builder, loc, generateExpr(expr->getCond()));
  auto target = expr->getType()->isPointerType()
                    ? types.convertStorageType(expr->getType())
                    : types.convertType(expr->getType());
  if (!condition || !target) {
    return {};
  }
  bool hasResult = !llvm::isa<mlir::NoneType>(target);
  auto selection = mlir::scf::IfOp::create(
      builder, loc, hasResult ? mlir::TypeRange{target} : mlir::TypeRange{},
      condition, true);
  for (auto [block, arm] :
       {std::pair{selection.thenBlock(), expr->getTrueExpr()},
        std::pair{selection.elseBlock(), expr->getFalseExpr()}}) {
    mlir::OpBuilder::InsertionGuard guard(builder);
    if (!block->empty() && llvm::isa<mlir::scf::YieldOp>(block->back())) {
      block->back().erase();
    }
    builder.setInsertionPointToStart(block);
    mlir::Value value;
    if (hasResult) {
      value = generateExpr(arm);
    } else if (mlir::failed(emitDiscarded(*arm))) {
      return {};
    }
    if (context.source().failed()) {
      return {};
    }
    if (hasResult) {
      value = clang_mlir::toValue(
          builder, loc, value, target,
          arm->getType()->isSignedIntegerOrEnumerationType());
      if (!value) {
        context.source().error(arm->getExprLoc(),
                               "cannot convert conditional result");
        return {};
      }
    }
    mlir::scf::YieldOp::create(
        builder, loc, hasResult ? mlir::ValueRange{value} : mlir::ValueRange{});
  }
  builder.setInsertionPointAfter(selection);
  return hasResult ? selection.getResult(0) : mlir::Value{};
}

} // namespace cmlir::frontend
