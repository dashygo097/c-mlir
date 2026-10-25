#include "../ExpressionLowering.h"
#include "../PointerArithmetic.h"
#include "clang-mlir/Casts.h"
#include "clang-mlir/Constants.h"
#include "clang-mlir/FloatingPoint.h"
#include "clang-mlir/Numerics.h"

namespace cmlir::frontend {

auto ExpressionLowering::emitIncDecLValue(clang::UnaryOperator &expr,
                                          mlir::Type resultType,
                                          LValueConsumer consume)
    -> mlir::FailureOr<mlir::Value> {
  return emitLValue(
      expr.getSubExpr(), resultType,
      [&](const LValueResult &result) -> mlir::FailureOr<mlir::Value> {
        const auto *object = std::get_if<ObjectResult>(&result);
        if (!object) {
          return mlir::failure();
        }
        auto value = updateIncDec(&expr, object->address);
        if (!value) {
          return mlir::failure();
        }
        return consume(ObjectResult{object->address, value});
      });
}

auto ExpressionLowering::generateIncDecUnaryOperator(clang::UnaryOperator *expr)
    -> mlir::Value {
  return loadLValue(expr);
}

auto ExpressionLowering::updateIncDec(clang::UnaryOperator *expr,
                                      const clang_mlir::Address &address)
    -> mlir::Value {
  auto &builder = context.builder();
  auto loc = context.source().location(expr->getExprLoc());
  auto old = address.load(builder, loc);
  if (!old) {
    return {};
  }
  mlir::Value value;
  if (expr->getSubExpr()->getType()->isPointerType()) {
    auto element = types.convertStorageType(
        expr->getSubExpr()->getType()->getPointeeType());
    if (!element) {
      return {};
    }
    value = emitPointerOffset(builder, loc, old,
                              clang_mlir::indexConst(builder, loc, 1), element,
                              /*signedOffset=*/true, expr->isDecrementOp());
    value = clang_mlir::toValue(builder, loc, value, address.elementType());
  } else {
    auto flags =
        clang_mlir::getFastMathFlags(*expr, context.ast().getLangOpts());
    value = expr->isIncrementOp() ? clang_mlir::inc(builder, loc, old, flags)
                                  : clang_mlir::dec(builder, loc, old, flags);
  }
  if (!value || mlir::failed(address.store(builder, loc, value))) {
    return {};
  }
  return expr->isPrefix() ? value : old;
}

} // namespace cmlir::frontend
