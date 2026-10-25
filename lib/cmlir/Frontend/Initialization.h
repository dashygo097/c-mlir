#ifndef CMLIR_FRONTEND_INITIALIZATION_H
#define CMLIR_FRONTEND_INITIALIZATION_H

#include "Types/TypeLowering.h"
#include "clang/AST/Expr.h"

namespace cmlir::frontend {

class ExpressionLowering;

class ObjectInitialization {
public:
  ObjectInitialization(clang_mlir::Context &context, TypeLowering &types,
                       ExpressionLowering &expressions)
      : context(context), types(types), expressions(expressions) {}

  auto emit(clang::QualType type, clang::Expr *initializer, mlir::Value storage,
            llvm::ArrayRef<mlir::Value> indices = {}) -> bool;

private:
  clang_mlir::Context &context;
  TypeLowering &types;
  ExpressionLowering &expressions;
};

} // namespace cmlir::frontend

#endif // CMLIR_FRONTEND_INITIALIZATION_H
