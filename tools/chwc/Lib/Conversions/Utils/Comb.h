#ifndef CHWC_UTILS_COMB_H
#define CHWC_UTILS_COMB_H

#include "./Builder.h"
#include "./Casts.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/WithColor.h"

namespace chwc::utils {

inline auto sameTypeBinary(mlir::OpBuilder &builder, mlir::Location loc,
                           llvm::StringRef opName, mlir::Value lhs,
                           mlir::Value rhs) -> mlir::Value {
  if (!lhs || !rhs) {
    return nullptr;
  }

  if (lhs.getType() != rhs.getType()) {
    llvm::WithColor::error()
        << "chwc: binary operands must have the same type\n";
    return nullptr;
  }

  llvm::SmallVector<mlir::Value, 2> operands{lhs, rhs};
  return createOneResultOp(builder, loc, opName, operands, lhs.getType());
}

inline auto add(mlir::OpBuilder &builder, mlir::Location loc, mlir::Value lhs,
                mlir::Value rhs) -> mlir::Value {
  return sameTypeBinary(builder, loc, "comb.add", lhs, rhs);
}

inline auto sub(mlir::OpBuilder &builder, mlir::Location loc, mlir::Value lhs,
                mlir::Value rhs) -> mlir::Value {
  return sameTypeBinary(builder, loc, "comb.sub", lhs, rhs);
}

inline auto mul(mlir::OpBuilder &builder, mlir::Location loc, mlir::Value lhs,
                mlir::Value rhs) -> mlir::Value {
  return sameTypeBinary(builder, loc, "comb.mul", lhs, rhs);
}

inline auto bitAnd(mlir::OpBuilder &builder, mlir::Location loc,
                   mlir::Value lhs, mlir::Value rhs) -> mlir::Value {
  return sameTypeBinary(builder, loc, "comb.and", lhs, rhs);
}

inline auto bitOr(mlir::OpBuilder &builder, mlir::Location loc, mlir::Value lhs,
                  mlir::Value rhs) -> mlir::Value {
  return sameTypeBinary(builder, loc, "comb.or", lhs, rhs);
}

inline auto bitXor(mlir::OpBuilder &builder, mlir::Location loc,
                   mlir::Value lhs, mlir::Value rhs) -> mlir::Value {
  return sameTypeBinary(builder, loc, "comb.xor", lhs, rhs);
}

inline auto icmpEq(mlir::OpBuilder &builder, mlir::Location loc,
                   mlir::Value lhs, mlir::Value rhs) -> mlir::Value {
  if (!lhs || !rhs) {
    return nullptr;
  }

  return circt::comb::ICmpOp::create(builder, loc,
                                     circt::comb::ICmpPredicate::eq, lhs, rhs)
      .getResult();
}

inline auto icmpNe(mlir::OpBuilder &builder, mlir::Location loc,
                   mlir::Value lhs, mlir::Value rhs) -> mlir::Value {
  if (!lhs || !rhs) {
    return nullptr;
  }

  return circt::comb::ICmpOp::create(builder, loc,
                                     circt::comb::ICmpPredicate::ne, lhs, rhs)
      .getResult();
}

inline auto icmpSlt(mlir::OpBuilder &builder, mlir::Location loc,
                    mlir::Value lhs, mlir::Value rhs) -> mlir::Value {
  if (!lhs || !rhs) {
    return nullptr;
  }

  return circt::comb::ICmpOp::create(builder, loc,
                                     circt::comb::ICmpPredicate::slt, lhs, rhs)
      .getResult();
}

inline auto icmpSle(mlir::OpBuilder &builder, mlir::Location loc,
                    mlir::Value lhs, mlir::Value rhs) -> mlir::Value {
  if (!lhs || !rhs) {
    return nullptr;
  }

  return circt::comb::ICmpOp::create(builder, loc,
                                     circt::comb::ICmpPredicate::sle, lhs, rhs)
      .getResult();
}

inline auto icmpSgt(mlir::OpBuilder &builder, mlir::Location loc,
                    mlir::Value lhs, mlir::Value rhs) -> mlir::Value {
  if (!lhs || !rhs) {
    return nullptr;
  }

  return circt::comb::ICmpOp::create(builder, loc,
                                     circt::comb::ICmpPredicate::sgt, lhs, rhs)
      .getResult();
}

inline auto icmpSge(mlir::OpBuilder &builder, mlir::Location loc,
                    mlir::Value lhs, mlir::Value rhs) -> mlir::Value {
  if (!lhs || !rhs) {
    return nullptr;
  }

  return circt::comb::ICmpOp::create(builder, loc,
                                     circt::comb::ICmpPredicate::sge, lhs, rhs)
      .getResult();
}

inline auto mux(mlir::OpBuilder &builder, mlir::Location loc, mlir::Value cond,
                mlir::Value trueValue, mlir::Value falseValue) -> mlir::Value {
  if (!cond || !trueValue || !falseValue) {
    return nullptr;
  }

  cond = toBool(builder, loc, cond);
  if (!cond) {
    return nullptr;
  }

  if (trueValue.getType() != falseValue.getType()) {
    llvm::WithColor::error() << "chwc: mux operands must have the same type\n";
    return nullptr;
  }

  llvm::SmallVector<mlir::Value, 3> operands{cond, trueValue, falseValue};
  return createOneResultOp(builder, loc, "comb.mux", operands,
                           trueValue.getType());
}

} // namespace chwc::utils

#endif // CHWC_UTILS_COMB_H
