#ifndef CMLIR_FRONTEND_MATH_CALL_H
#define CMLIR_FRONTEND_MATH_CALL_H

#include "CallLowering.h"

namespace cmlir::frontend {

enum class MathDomain { Integer, Float };

struct MathIntrinsic {
  llvm::StringLiteral names;
  mlir::Value (*emit)(mlir::OpBuilder &, mlir::Location,
                      llvm::ArrayRef<mlir::Value>);
  unsigned arity;
  MathDomain domain;
  bool predicate;
};

auto findMathIntrinsic(const clang::FunctionDecl &, mlir::FunctionType)
    -> const MathIntrinsic *;

} // namespace cmlir::frontend

#endif // CMLIR_FRONTEND_MATH_CALL_H
