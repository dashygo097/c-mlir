#ifndef CHWC_UTILS_HWOPS_H
#define CHWC_UTILS_HWOPS_H

#include "../../Converter.h"

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/PortImplementation.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/WithColor.h"

namespace chwc::utils {

inline circt::hw::HWModuleOp currentHWModule;
inline llvm::DenseMap<const clang::FieldDecl *, mlir::Value> inputValueTable;
inline llvm::SmallVector<mlir::Value, 8> outputValues;

inline auto createOneResultOp(mlir::OpBuilder &builder, mlir::Location loc,
                              llvm::StringRef opName,
                              llvm::ArrayRef<mlir::Value> operands,
                              mlir::Type resultType) -> mlir::Value {
  mlir::OperationState state(loc, opName);
  state.addOperands(operands);
  state.addTypes(resultType);
  return builder.create(state)->getResult(0);
}

inline auto createSameTypeBinaryOp(mlir::OpBuilder &builder, mlir::Location loc,
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

inline auto intConst(mlir::OpBuilder &builder, mlir::Location loc,
                     mlir::Type type, uint64_t value) -> mlir::Value {
  auto intType = mlir::dyn_cast<mlir::IntegerType>(type);
  if (!intType) {
    llvm::WithColor::error()
        << "chwc: hw.constant requires integer result type\n";
    return nullptr;
  }

  return circt::hw::ConstantOp::create(builder, loc, type, value).getResult();
}

inline auto zeroValue(mlir::OpBuilder &builder, mlir::Location loc,
                      mlir::Type type) -> mlir::Value {
  return intConst(builder, loc, type, 0);
}

inline auto oneValue(mlir::OpBuilder &builder, mlir::Location loc,
                     mlir::Type type) -> mlir::Value {
  return intConst(builder, loc, type, 1);
}

inline auto toBool(mlir::OpBuilder &builder, mlir::Location loc,
                   mlir::Value value) -> mlir::Value {
  if (!value) {
    return nullptr;
  }

  auto intType = mlir::dyn_cast<mlir::IntegerType>(value.getType());
  if (!intType) {
    llvm::WithColor::error()
        << "chwc: only integer value can be converted to bool\n";
    return nullptr;
  }

  if (intType.getWidth() == 1) {
    return value;
  }

  mlir::Value zero = zeroValue(builder, loc, value.getType());
  if (!zero) {
    return nullptr;
  }

  return circt::comb::ICmpOp::create(
             builder, loc, circt::comb::ICmpPredicate::ne, value, zero)
      .getResult();
}

inline auto add(mlir::OpBuilder &builder, mlir::Location loc, mlir::Value lhs,
                mlir::Value rhs) -> mlir::Value {
  return createSameTypeBinaryOp(builder, loc, "comb.add", lhs, rhs);
}

inline auto sub(mlir::OpBuilder &builder, mlir::Location loc, mlir::Value lhs,
                mlir::Value rhs) -> mlir::Value {
  return createSameTypeBinaryOp(builder, loc, "comb.sub", lhs, rhs);
}

inline auto mul(mlir::OpBuilder &builder, mlir::Location loc, mlir::Value lhs,
                mlir::Value rhs) -> mlir::Value {
  return createSameTypeBinaryOp(builder, loc, "comb.mul", lhs, rhs);
}

inline auto bitAnd(mlir::OpBuilder &builder, mlir::Location loc,
                   mlir::Value lhs, mlir::Value rhs) -> mlir::Value {
  return createSameTypeBinaryOp(builder, loc, "comb.and", lhs, rhs);
}

inline auto bitOr(mlir::OpBuilder &builder, mlir::Location loc, mlir::Value lhs,
                  mlir::Value rhs) -> mlir::Value {
  return createSameTypeBinaryOp(builder, loc, "comb.or", lhs, rhs);
}

inline auto bitXor(mlir::OpBuilder &builder, mlir::Location loc,
                   mlir::Value lhs, mlir::Value rhs) -> mlir::Value {
  return createSameTypeBinaryOp(builder, loc, "comb.xor", lhs, rhs);
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

inline void beginHWModule(
    mlir::OpBuilder &builder, mlir::Location loc,
    clang::CXXRecordDecl *recordDecl,
    llvm::DenseMap<const clang::FieldDecl *, HWFieldInfo> &fieldTable) {
  inputValueTable.clear();
  outputValues.clear();

  llvm::SmallVector<circt::hw::PortInfo, 8> ports;

  size_t inputArgNo = 0;
  size_t outputArgNo = 0;

  for (auto &[fieldDecl, fieldInfo] : fieldTable) {
    if (fieldInfo.kind == HWFieldKind::Input) {
      circt::hw::PortInfo port;
      port.name = builder.getStringAttr(fieldInfo.name);
      port.type = fieldInfo.type;
      port.dir = circt::hw::ModulePort::Direction::Input;
      port.argNum = inputArgNo++;
      port.loc = loc;
      ports.push_back(port);
      continue;
    }

    if (fieldInfo.kind == HWFieldKind::Output) {
      circt::hw::PortInfo port;
      port.name = builder.getStringAttr(fieldInfo.name);
      port.type = fieldInfo.type;
      port.dir = circt::hw::ModulePort::Direction::Output;
      port.argNum = outputArgNo++;
      port.loc = loc;
      ports.push_back(port);
      continue;
    }
  }

  currentHWModule = circt::hw::HWModuleOp::create(
      builder, loc, builder.getStringAttr(recordDecl->getNameAsString()),
      ports);

  mlir::Block *bodyBlock = currentHWModule.getBodyBlock();

  if (!bodyBlock->empty()) {
    if (auto outputOp =
            mlir::dyn_cast<circt::hw::OutputOp>(bodyBlock->back())) {
      outputOp.erase();
    }
  }

  size_t argIndex = 0;
  for (auto &[fieldDecl, fieldInfo] : fieldTable) {
    if (fieldInfo.kind != HWFieldKind::Input) {
      continue;
    }

    if (argIndex >= bodyBlock->getNumArguments()) {
      llvm::WithColor::error()
          << "chwc: internal error: hw.module input block arg missing\n";
      continue;
    }

    inputValueTable[fieldDecl] = bodyBlock->getArgument(argIndex++);
  }

  builder.setInsertionPointToEnd(bodyBlock);
}

inline void endHWModule(mlir::OpBuilder &builder, mlir::Location loc) {
  if (!currentHWModule) {
    return;
  }

  circt::hw::OutputOp::create(builder, loc, outputValues);

  builder.setInsertionPointAfter(currentHWModule);

  currentHWModule = nullptr;
  inputValueTable.clear();
  outputValues.clear();
}

inline auto getInputValue(mlir::OpBuilder &builder, mlir::Location loc,
                          const HWFieldInfo &fieldInfo) -> mlir::Value {
  (void)builder;
  (void)loc;

  mlir::Value value = inputValueTable.lookup(fieldInfo.fieldDecl);
  if (!value) {
    llvm::WithColor::error()
        << "chwc: input port value is not wired: " << fieldInfo.name << "\n";
  }

  return value;
}

inline auto emitRegister(mlir::OpBuilder &builder, mlir::Location loc,
                         const HWFieldInfo &fieldInfo, mlir::Value resetValue)
    -> mlir::Value {
  (void)builder;
  (void)loc;
  (void)fieldInfo;

  return resetValue;
}

inline void emitRegisterNextAssign(mlir::OpBuilder &builder, mlir::Location loc,
                                   const HWFieldInfo &fieldInfo,
                                   mlir::Value nextValue) {
  (void)builder;
  (void)loc;
  (void)fieldInfo;
  (void)nextValue;
}

inline void emitOutputAssign(mlir::OpBuilder &builder, mlir::Location loc,
                             const HWFieldInfo &fieldInfo, mlir::Value value) {
  (void)builder;
  (void)loc;
  (void)fieldInfo;

  if (!value) {
    llvm::WithColor::error()
        << "chwc: null output value for " << fieldInfo.name << "\n";
    return;
  }

  outputValues.push_back(value);
}

} // namespace chwc::utils

#endif // CHWC_UTILS_HWOPS_H
