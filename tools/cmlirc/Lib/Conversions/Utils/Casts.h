#ifndef CMLIRC_CASTS_H
#define CMLIRC_CASTS_H

#include "./Constants.h"

namespace cmlirc::detail {
inline mlir::Value toIndex(mlir::OpBuilder &builder, mlir::Location loc,
                           mlir::Value value) {
  if (value.getType().isIndex())
    return value;

  if (mlir::isa<mlir::FloatType>(value.getType())) {
    value =
        mlir::arith::FPToSIOp::create(builder, loc, builder.getI64Type(), value)
            .getResult();
  }

  return mlir::arith::IndexCastOp::create(builder, loc, builder.getIndexType(),
                                          value)
      .getResult();
}

inline mlir::Value toBool(mlir::OpBuilder &builder, mlir::Location loc,
                          mlir::Value value) {
  mlir::Type type = value.getType();

  if (type.isInteger(1))
    return value;

  if (auto intType = mlir::dyn_cast<mlir::IntegerType>(type)) {
    return mlir::arith::CmpIOp::create(builder, loc,
                                       mlir::arith::CmpIPredicate::ne, value,
                                       detail::intConst(builder, loc, type, 0))
        .getResult();
  }

  if (auto floatType = mlir::dyn_cast<mlir::FloatType>(type)) {
    return mlir::arith::CmpFOp::create(
               builder, loc,
               mlir::arith::CmpFPredicate::ONE, // ONE = Ordered Not Equal
               value, detail::floatConst(builder, loc, type, 0.0))
        .getResult();
  }

  return nullptr;
}

inline std::optional<int64_t> getInt(mlir::Value value) {
  while (auto cast = value.getDefiningOp<mlir::arith::IndexCastOp>())
    value = cast.getIn();

  if (auto c = value.getDefiningOp<mlir::arith::ConstantOp>())
    if (auto ia = mlir::dyn_cast<mlir::IntegerAttr>(c.getValue()))
      return ia.getInt();

  return std::nullopt;
}

inline std::optional<double> getFloat(mlir::Value value) {
  while (auto cast = value.getDefiningOp<mlir::arith::IndexCastOp>())
    value = cast.getIn();

  if (auto c = value.getDefiningOp<mlir::arith::ConstantOp>())
    if (auto fa = mlir::dyn_cast<mlir::FloatAttr>(c.getValue()))
      return fa.getValueAsDouble();

  return std::nullopt;
}

inline mlir::Value promoteValue(mlir::OpBuilder &b, mlir::Location loc,
                                mlir::Value value, mlir::Type targetType,
                                bool isSigned) {
  mlir::Type srcType = value.getType();
  if (srcType == targetType)
    return value;

  auto srcInt = mlir::dyn_cast<mlir::IntegerType>(srcType);
  auto dstInt = mlir::dyn_cast<mlir::IntegerType>(targetType);
  auto srcFlt = mlir::dyn_cast<mlir::FloatType>(srcType);
  auto dstFlt = mlir::dyn_cast<mlir::FloatType>(targetType);

  // int → wider int
  if (srcInt && dstInt && srcInt.getWidth() < dstInt.getWidth())
    return isSigned ? mlir::arith::ExtSIOp::create(b, loc, targetType, value)
                          .getResult()
                    : mlir::arith::ExtUIOp::create(b, loc, targetType, value)
                          .getResult();

  // float → wider float
  if (srcFlt && dstFlt && srcFlt.getWidth() < dstFlt.getWidth())
    return mlir::arith::ExtFOp::create(b, loc, targetType, value).getResult();

  // int → float
  if (srcInt && dstFlt)
    return isSigned ? mlir::arith::SIToFPOp::create(b, loc, targetType, value)
                          .getResult()
                    : mlir::arith::UIToFPOp::create(b, loc, targetType, value)
                          .getResult();

  return value;
}

inline mlir::Value truncateValue(mlir::OpBuilder &b, mlir::Location loc,
                                 mlir::Value value, mlir::Type targetType) {
  mlir::Type srcType = value.getType();
  if (srcType == targetType)
    return value;

  auto srcInt = mlir::dyn_cast<mlir::IntegerType>(srcType);
  auto dstInt = mlir::dyn_cast<mlir::IntegerType>(targetType);
  auto srcFlt = mlir::dyn_cast<mlir::FloatType>(srcType);
  auto dstFlt = mlir::dyn_cast<mlir::FloatType>(targetType);

  // wider int → int
  if (srcInt && dstInt && srcInt.getWidth() > dstInt.getWidth())
    return mlir::arith::TruncIOp::create(b, loc, targetType, value).getResult();

  // wider float → float
  if (srcFlt && dstFlt && srcFlt.getWidth() > dstFlt.getWidth())
    return mlir::arith::TruncFOp::create(b, loc, targetType, value).getResult();

  return value;
}

} // namespace cmlirc::detail

#endif // CMLIRC_CASTS_H
