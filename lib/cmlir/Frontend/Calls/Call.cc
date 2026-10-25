#include "../Expressions/ExpressionLowering.h"
#include "CallLowering.h"
#include "clang-mlir/FloatingPoint.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

namespace cmlir::frontend {

auto ExpressionLowering::generateCallExpr(clang::CallExpr *expr)
    -> mlir::FailureOr<mlir::Value> {
  const auto *callee = expr->getDirectCallee();
  if (!callee) {
    context.source().error(expr->getExprLoc(),
                           "indirect calls are unsupported");
    return mlir::failure();
  }
  if (auto *method = llvm::dyn_cast<clang::CXXMethodDecl>(callee)) {
    if (method->isVirtual() || llvm::isa<clang::CXXDestructorDecl>(method)) {
      context.source().error(
          expr->getExprLoc(),
          "virtual calls and destructor calls are unsupported");
      return mlir::failure();
    }
  }

  llvm::SmallVector<CallArg, 8> arguments;
  if (auto *memberCall = llvm::dyn_cast<clang::CXXMemberCallExpr>(expr)) {
    auto *object = memberCall->getImplicitObjectArgument();
    mlir::Value value;
    if (object->isGLValue() && !object->getType()->isPointerType()) {
      auto address = addressOf(object);
      if (mlir::failed(address)) {
        return mlir::failure();
      }
      value = address->materialize(
          context.builder(), context.source().location(expr->getExprLoc()));
    } else {
      value = generateExpr(object);
    }
    if (!value) {
      return mlir::failure();
    }
    // Materialize a record value only when the implicit object is an rvalue.
    // Addressable objects already carry their storage pointer.
    if (mlir::isa<mlir::LLVM::LLVMStructType>(value.getType())) {
      auto &builder = context.builder();
      auto loc = context.source().location(expr->getExprLoc());
      value = clang_mlir::createLocalSlot(builder, loc, value);
    }
    arguments.push_back({value, object->getType()});
  }
  for (unsigned i = 0; i < expr->getNumArgs(); ++i) {
    auto *argument = expr->getArg(i);
    mlir::Value value;
    if (i < callee->getNumParams() &&
        callee->getParamDecl(i)->getType()->isReferenceType()) {
      auto address = addressOf(argument);
      if (mlir::failed(address)) {
        return mlir::failure();
      }
      value = address->materialize(
          context.builder(), context.source().location(argument->getExprLoc()));
    } else {
      value = generateExpr(argument);
    }
    if (!value) {
      return mlir::failure();
    }
    arguments.push_back({value, argument->getType()});
  }
  auto result = calls.emit(*callee, expr->getExprLoc(), arguments);
  if (mlir::failed(result)) {
    if (!context.source().failed()) {
      context.source().error(expr->getExprLoc(),
                             "function call lowering failed");
    }
    return mlir::failure();
  }
  if (*result) {
    if (auto operation =
            mlir::dyn_cast_if_present<mlir::arith::ArithFastMathInterface>(
                (*result).getDefiningOp())) {
      auto flags =
          clang_mlir::getFastMathFlags(*expr, context.ast().getLangOpts());
      operation->setAttr(
          operation.getFastMathAttrName(),
          mlir::arith::FastMathFlagsAttr::get(&context.ir(), flags));
    }
  }
  return result;
}

} // namespace cmlir::frontend
