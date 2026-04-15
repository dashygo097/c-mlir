#ifndef CMLIRC_CASTS_H
#define CMLIRC_CASTS_H

#include "./Constants.h"

namespace cmlirc::utils {
inline auto toIndex(mlir::OpBuilder &builder, mlir::Location loc,
                    mlir::Value value) -> mlir::Value {
  if (value.getType().isIndex()) {
    return value;
  }

  if (mlir::isa<mlir::FloatType>(value.getType())) {
    value =
        mlir::arith::FPToSIOp::create(builder, loc, builder.getI64Type(), value)
            .getResult();
  }

  return mlir::arith::IndexCastOp::create(builder, loc, builder.getIndexType(),
                                          value)
      .getResult();
}

inline auto toBool(mlir::OpBuilder &builder, mlir::Location loc,
                   mlir::Value value) -> mlir::Value {
  mlir::Type type = value.getType();
  if (type.isInteger(1)) {
    return value;
  }
  if (auto intType = mlir::dyn_cast<mlir::IntegerType>(type)) {
    return mlir::arith::CmpIOp::create(builder, loc,
                                       mlir::arith::CmpIPredicate::ne, value,
                                       utils::intConst(builder, loc, type, 0))
        .getResult();
  }
  if (auto floatType = mlir::dyn_cast<mlir::FloatType>(type)) {
    return mlir::arith::CmpFOp::create(
               builder, loc,
               mlir::arith::CmpFPredicate::ONE, // ONE = Ordered Not Equal
               value, utils::floatConst(builder, loc, type, 0.0))
        .getResult();
  }
  return nullptr;
}

inline auto getInt(mlir::Value value) -> std::optional<int64_t> {
  while (auto cast = value.getDefiningOp<mlir::arith::IndexCastOp>()) {
    value = cast.getIn();
  }

  if (auto c = value.getDefiningOp<mlir::arith::ConstantOp>()) {
    if (auto ia = mlir::dyn_cast<mlir::IntegerAttr>(c.getValue())) {
      return ia.getInt();
    }
  }

  return std::nullopt;
}

inline auto getFloat(mlir::Value value) -> std::optional<double> {
  while (auto cast = value.getDefiningOp<mlir::arith::IndexCastOp>()) {
    value = cast.getIn();
  }

  if (auto c = value.getDefiningOp<mlir::arith::ConstantOp>()) {
    if (auto fa = mlir::dyn_cast<mlir::FloatAttr>(c.getValue())) {
      return fa.getValueAsDouble();
    }
  }

  return std::nullopt;
}

inline auto promoteValue(mlir::OpBuilder &builder, mlir::Location loc,
                         mlir::Value value, mlir::Type targetType,
                         bool isSigned) -> mlir::Value {
  mlir::Type srcType = value.getType();
  if (srcType == targetType) {
    return value;
  }

  auto srcInt = mlir::dyn_cast<mlir::IntegerType>(srcType);
  auto dstInt = mlir::dyn_cast<mlir::IntegerType>(targetType);
  auto srcFlt = mlir::dyn_cast<mlir::FloatType>(srcType);
  auto dstFlt = mlir::dyn_cast<mlir::FloatType>(targetType);

  // int → wider int
  if (srcInt && dstInt && srcInt.getWidth() < dstInt.getWidth()) {
    return isSigned
               ? mlir::arith::ExtSIOp::create(builder, loc, targetType, value)
                     .getResult()
               : mlir::arith::ExtUIOp::create(builder, loc, targetType, value)
                     .getResult();
  }

  // float → wider float
  if (srcFlt && dstFlt && srcFlt.getWidth() < dstFlt.getWidth()) {
    return mlir::arith::ExtFOp::create(builder, loc, targetType, value)
        .getResult();
  }

  // int → float
  if (srcInt && dstFlt) {
    return isSigned
               ? mlir::arith::SIToFPOp::create(builder, loc, targetType, value)
                     .getResult()
               : mlir::arith::UIToFPOp::create(builder, loc, targetType, value)
                     .getResult();
  }

  return value;
}

inline auto truncateValue(mlir::OpBuilder &builder, mlir::Location loc,
                          mlir::Value value, mlir::Type targetType,
                          bool isSigned = true) -> mlir::Value {
  mlir::Type srcType = value.getType();
  if (srcType == targetType) {
    return value;
  }

  auto srcInt = mlir::dyn_cast<mlir::IntegerType>(srcType);
  auto dstInt = mlir::dyn_cast<mlir::IntegerType>(targetType);
  auto srcFlt = mlir::dyn_cast<mlir::FloatType>(srcType);
  auto dstFlt = mlir::dyn_cast<mlir::FloatType>(targetType);

  // int → narrower int
  if (srcInt && dstInt && srcInt.getWidth() > dstInt.getWidth()) {
    return mlir::arith::TruncIOp::create(builder, loc, targetType, value)
        .getResult();
  }

  // float → narrower float
  if (srcFlt && dstFlt && srcFlt.getWidth() > dstFlt.getWidth()) {
    return mlir::arith::TruncFOp::create(builder, loc, targetType, value)
        .getResult();
  }

  //  float → int
  if (srcFlt && dstInt) {
    return isSigned
               ? mlir::arith::FPToSIOp::create(builder, loc, targetType, value)
                     .getResult()
               : mlir::arith::FPToUIOp::create(builder, loc, targetType, value)
                     .getResult();
  }

  // int → float
  if (srcInt && dstFlt) {
    return isSigned
               ? mlir::arith::SIToFPOp::create(builder, loc, targetType, value)
                     .getResult()
               : mlir::arith::UIToFPOp::create(builder, loc, targetType, value)
                     .getResult();
  }

  return value;
}

inline auto castValue(mlir::OpBuilder &builder, mlir::Location loc,
                      mlir::Value value, mlir::Type targetType, bool isSigned)
    -> mlir::Value {
  mlir::Type srcType = value.getType();
  if (srcType == targetType) {
    return value;
  }
  mlir::Value promoted =
      promoteValue(builder, loc, value, targetType, isSigned);
  return truncateValue(builder, loc, promoted, targetType, isSigned);
}

} // namespace cmlirc::utils

#endif // CMLIRC_CASTS_H
