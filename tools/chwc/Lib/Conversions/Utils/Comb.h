#ifndef CHWC_UTILS_COMB_H
#define CHWC_UTILS_COMB_H

#include "./Constant.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"
#include "llvm/Support/WithColor.h"

namespace chwc::utils {

inline auto sameTypeBinaryOp(mlir::OpBuilder &builder, mlir::Location loc,
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

  mlir::OperationState state(loc, opName);
  state.addOperands({lhs, rhs});
  state.addTypes(lhs.getType());

  mlir::Operation *op = builder.create(state);
  return op->getResult(0);
}

inline auto add(mlir::OpBuilder &builder, mlir::Location loc, mlir::Value lhs,
                mlir::Value rhs) -> mlir::Value {
  return sameTypeBinaryOp(builder, loc, "comb.add", lhs, rhs);
}

inline auto sub(mlir::OpBuilder &builder, mlir::Location loc, mlir::Value lhs,
                mlir::Value rhs) -> mlir::Value {
  return sameTypeBinaryOp(builder, loc, "comb.sub", lhs, rhs);
}

inline auto mul(mlir::OpBuilder &builder, mlir::Location loc, mlir::Value lhs,
                mlir::Value rhs) -> mlir::Value {
  return sameTypeBinaryOp(builder, loc, "comb.mul", lhs, rhs);
}

inline auto divU(mlir::OpBuilder &builder, mlir::Location loc, mlir::Value lhs,
                 mlir::Value rhs) -> mlir::Value {
  return sameTypeBinaryOp(builder, loc, "comb.divu", lhs, rhs);
}

inline auto divS(mlir::OpBuilder &builder, mlir::Location loc, mlir::Value lhs,
                 mlir::Value rhs) -> mlir::Value {
  return sameTypeBinaryOp(builder, loc, "comb.divs", lhs, rhs);
}

inline auto modU(mlir::OpBuilder &builder, mlir::Location loc, mlir::Value lhs,
                 mlir::Value rhs) -> mlir::Value {
  return sameTypeBinaryOp(builder, loc, "comb.modu", lhs, rhs);
}

inline auto modS(mlir::OpBuilder &builder, mlir::Location loc, mlir::Value lhs,
                 mlir::Value rhs) -> mlir::Value {
  return sameTypeBinaryOp(builder, loc, "comb.mods", lhs, rhs);
}

inline auto bitAnd(mlir::OpBuilder &builder, mlir::Location loc,
                   mlir::Value lhs, mlir::Value rhs) -> mlir::Value {
  return sameTypeBinaryOp(builder, loc, "comb.and", lhs, rhs);
}

inline auto bitOr(mlir::OpBuilder &builder, mlir::Location loc, mlir::Value lhs,
                  mlir::Value rhs) -> mlir::Value {
  return sameTypeBinaryOp(builder, loc, "comb.or", lhs, rhs);
}

inline auto bitXor(mlir::OpBuilder &builder, mlir::Location loc,
                   mlir::Value lhs, mlir::Value rhs) -> mlir::Value {
  return sameTypeBinaryOp(builder, loc, "comb.xor", lhs, rhs);
}

inline auto shl(mlir::OpBuilder &builder, mlir::Location loc, mlir::Value lhs,
                mlir::Value rhs) -> mlir::Value {
  return sameTypeBinaryOp(builder, loc, "comb.shl", lhs, rhs);
}

inline auto shrU(mlir::OpBuilder &builder, mlir::Location loc, mlir::Value lhs,
                 mlir::Value rhs) -> mlir::Value {
  return sameTypeBinaryOp(builder, loc, "comb.shru", lhs, rhs);
}

inline auto shrS(mlir::OpBuilder &builder, mlir::Location loc, mlir::Value lhs,
                 mlir::Value rhs) -> mlir::Value {
  return sameTypeBinaryOp(builder, loc, "comb.shrs", lhs, rhs);
}

inline auto icmp(mlir::OpBuilder &builder, mlir::Location loc,
                 llvm::StringRef predicate, mlir::Value lhs, mlir::Value rhs)
    -> mlir::Value {
  if (!lhs || !rhs) {
    return nullptr;
  }

  if (lhs.getType() != rhs.getType()) {
    llvm::WithColor::error() << "chwc: icmp operands must have the same type\n";
    return nullptr;
  }

  mlir::OperationState state(loc, "comb.icmp");
  state.addOperands({lhs, rhs});
  state.addAttribute("predicate", builder.getStringAttr(predicate));
  state.addTypes(builder.getI1Type());

  mlir::Operation *op = builder.create(state);
  return op->getResult(0);
}

inline auto icmpEq(mlir::OpBuilder &builder, mlir::Location loc,
                   mlir::Value lhs, mlir::Value rhs) -> mlir::Value {
  return icmp(builder, loc, "eq", lhs, rhs);
}

inline auto icmpNe(mlir::OpBuilder &builder, mlir::Location loc,
                   mlir::Value lhs, mlir::Value rhs) -> mlir::Value {
  return icmp(builder, loc, "ne", lhs, rhs);
}

inline auto icmpSlt(mlir::OpBuilder &builder, mlir::Location loc,
                    mlir::Value lhs, mlir::Value rhs) -> mlir::Value {
  return icmp(builder, loc, "slt", lhs, rhs);
}

inline auto icmpSle(mlir::OpBuilder &builder, mlir::Location loc,
                    mlir::Value lhs, mlir::Value rhs) -> mlir::Value {
  return icmp(builder, loc, "sle", lhs, rhs);
}

inline auto icmpSgt(mlir::OpBuilder &builder, mlir::Location loc,
                    mlir::Value lhs, mlir::Value rhs) -> mlir::Value {
  return icmp(builder, loc, "sgt", lhs, rhs);
}

inline auto icmpSge(mlir::OpBuilder &builder, mlir::Location loc,
                    mlir::Value lhs, mlir::Value rhs) -> mlir::Value {
  return icmp(builder, loc, "sge", lhs, rhs);
}

inline auto icmpUlt(mlir::OpBuilder &builder, mlir::Location loc,
                    mlir::Value lhs, mlir::Value rhs) -> mlir::Value {
  return icmp(builder, loc, "ult", lhs, rhs);
}

inline auto icmpUle(mlir::OpBuilder &builder, mlir::Location loc,
                    mlir::Value lhs, mlir::Value rhs) -> mlir::Value {
  return icmp(builder, loc, "ule", lhs, rhs);
}

inline auto icmpUgt(mlir::OpBuilder &builder, mlir::Location loc,
                    mlir::Value lhs, mlir::Value rhs) -> mlir::Value {
  return icmp(builder, loc, "ugt", lhs, rhs);
}

inline auto icmpUge(mlir::OpBuilder &builder, mlir::Location loc,
                    mlir::Value lhs, mlir::Value rhs) -> mlir::Value {
  return icmp(builder, loc, "uge", lhs, rhs);
}

inline auto mux(mlir::OpBuilder &builder, mlir::Location loc, mlir::Value cond,
                mlir::Value trueValue, mlir::Value falseValue) -> mlir::Value {
  if (!cond || !trueValue || !falseValue) {
    return nullptr;
  }

  if (trueValue.getType() != falseValue.getType()) {
    llvm::WithColor::error()
        << "chwc: mux true/false values must have the same type\n";
    return nullptr;
  }

  auto condType = mlir::dyn_cast<mlir::IntegerType>(cond.getType());
  if (!condType || condType.getWidth() != 1) {
    llvm::WithColor::error() << "chwc: mux condition must be i1\n";
    return nullptr;
  }

  mlir::OperationState state(loc, "comb.mux");
  state.addOperands({cond, trueValue, falseValue});
  state.addTypes(trueValue.getType());

  mlir::Operation *op = builder.create(state);
  return op->getResult(0);
}

} // namespace chwc::utils

#endif // CHWC_UTILS_COMB_H
