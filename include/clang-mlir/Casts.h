#ifndef CLANG_MLIR_CASTS_H
#define CLANG_MLIR_CASTS_H

#include "Constants.h"
#include "Types.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Value.h"

namespace clang_mlir {

auto memrefToPointer(mlir::OpBuilder &builder, mlir::Location loc,
                     mlir::Value value, mlir::Type targetType) -> mlir::Value;
auto integerToPointer(mlir::OpBuilder &builder, mlir::Location loc,
                      mlir::Value value, mlir::Type targetType) -> mlir::Value;
auto pointerToInteger(mlir::OpBuilder &builder, mlir::Location loc,
                      mlir::Value value, mlir::Type targetType) -> mlir::Value;
auto indexToInteger(mlir::OpBuilder &builder, mlir::Location loc,
                    mlir::Value value, mlir::IntegerType targetType)
    -> mlir::Value;
auto integerToIndex(mlir::OpBuilder &builder, mlir::Location loc,
                    mlir::Value value) -> mlir::Value;
auto indexToInteger64(mlir::OpBuilder &builder, mlir::Location loc,
                      mlir::Value value) -> mlir::Value;
auto memrefToInteger(mlir::OpBuilder &builder, mlir::Location loc,
                     mlir::Value value, mlir::IntegerType targetType)
    -> mlir::Value;
auto toInteger(mlir::OpBuilder &builder, mlir::Location loc, mlir::Value value,
               mlir::IntegerType targetType, bool isSigned = true)
    -> mlir::Value;
auto toIndex(mlir::OpBuilder &builder, mlir::Location loc, mlir::Value value,
             bool isSigned = true) -> mlir::Value;
auto integerToFloat(mlir::OpBuilder &builder, mlir::Location loc,
                    mlir::Value value, mlir::FloatType targetType,
                    bool isSigned = true) -> mlir::Value;
auto indexToFloat(mlir::OpBuilder &builder, mlir::Location loc,
                  mlir::Value value, mlir::FloatType targetType,
                  bool isSigned = true) -> mlir::Value;
auto floatToFloat(mlir::OpBuilder &builder, mlir::Location loc,
                  mlir::Value value, mlir::FloatType targetType) -> mlir::Value;
auto toFloat(mlir::OpBuilder &builder, mlir::Location loc, mlir::Value value,
             mlir::FloatType targetType, bool isSigned = true) -> mlir::Value;
auto toPointer(mlir::OpBuilder &builder, mlir::Location loc, mlir::Value value,
               mlir::Type targetType) -> mlir::Value;
auto toNullPointer(mlir::OpBuilder &builder, mlir::Location loc,
                   mlir::Type targetType) -> mlir::Value;
auto integerToBool(mlir::OpBuilder &builder, mlir::Location loc,
                   mlir::Value value) -> mlir::Value;
auto floatToBool(mlir::OpBuilder &builder, mlir::Location loc,
                 mlir::Value value) -> mlir::Value;
auto pointerToBool(mlir::OpBuilder &builder, mlir::Location loc,
                   mlir::Value value) -> mlir::Value;
auto memrefToBool(mlir::OpBuilder &builder, mlir::Location loc,
                  mlir::Value value) -> mlir::Value;
auto toBool(mlir::OpBuilder &builder, mlir::Location loc, mlir::Value value)
    -> mlir::Value;
auto toValue(mlir::OpBuilder &builder, mlir::Location loc, mlir::Value value,
             mlir::Type targetType, bool isSigned = true) -> mlir::Value;
auto scalarToBitcast(mlir::OpBuilder &builder, mlir::Location loc,
                     mlir::Value value, mlir::Type targetType) -> mlir::Value;
auto toBitcastValue(mlir::OpBuilder &builder, mlir::Location loc,
                    mlir::Value value, mlir::Type targetType,
                    bool isSigned = true) -> mlir::Value;

} // namespace clang_mlir

#endif // CLANG_MLIR_CASTS_H
