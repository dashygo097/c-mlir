#include "../Lowering.h"
#include "CallLowering.h"
#include "clang-mlir/FloatingPoint.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

namespace cmlir::frontend {

auto ASTLowering::generateCallExpr(clang::CallExpr *expr) -> mlir::Value {
  const auto *callee = expr->getDirectCallee();
  if (!callee) {
    context.source().error(expr->getExprLoc(),
                           "indirect calls are unsupported");
    return {};
  }
  if (auto *method = llvm::dyn_cast<clang::CXXMethodDecl>(callee)) {
    if (method->isVirtual() || llvm::isa<clang::CXXDestructorDecl>(method)) {
      context.source().error(
          expr->getExprLoc(),
          "virtual calls and destructor calls are unsupported");
      return {};
    }
  }

  llvm::SmallVector<CallArg, 8> arguments;
  if (auto *memberCall = llvm::dyn_cast<clang::CXXMemberCallExpr>(expr)) {
    auto *object = memberCall->getImplicitObjectArgument();
    auto value = generateExpr(object);
    if (!value) {
      return {};
    }
    // Materialize a record value only when the implicit object is an rvalue.
    // Addressable objects already carry their storage pointer.
    if (mlir::isa<mlir::LLVM::LLVMStructType>(value.getType())) {
      auto &builder = context.builder();
      auto loc = context.source().location(expr->getExprLoc());
      auto one = mlir::arith::ConstantIntOp::create(builder, loc, 1, 64);
      auto pointer = mlir::LLVM::AllocaOp::create(
          builder, loc, mlir::LLVM::LLVMPointerType::get(builder.getContext()),
          value.getType(), one);
      mlir::LLVM::StoreOp::create(builder, loc, value, pointer);
      value = pointer;
    }
    arguments.push_back({value, object->getType()});
  }
  for (auto *argument : expr->arguments()) {
    auto value = generateExpr(argument);
    if (!value) {
      return {};
    }
    arguments.push_back({value, argument->getType()});
  }
  auto result = calls.emit(*callee, expr->getExprLoc(), arguments);
  if (mlir::failed(result)) {
    if (!context.source().failed()) {
      context.source().error(expr->getExprLoc(),
                             "function call lowering failed");
    }
    return {};
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
  return *result;
}

} // namespace cmlir::frontend
