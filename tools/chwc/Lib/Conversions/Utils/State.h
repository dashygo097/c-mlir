#ifndef CHWC_UTILS_STATE_H
#define CHWC_UTILS_STATE_H

#include "../../Converter.h"
#include "mlir/IR/Builders.h"

namespace chwc::utils {

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

} // namespace chwc::utils

#endif // CHWC_UTILS_STATE_H
