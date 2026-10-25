#include "ExpressionLowering.h"
#include "clang-mlir/Casts.h"
#include "clang-mlir/Constants.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

namespace cmlir::frontend {

auto ExpressionLowering::emitStorageAddress(clang::Expr *expr)
    -> mlir::FailureOr<clang_mlir::Address> {
  auto &builder = context.builder();
  auto loc = context.source().location(expr->getExprLoc());
  if (auto *temporary = llvm::dyn_cast<clang::MaterializeTemporaryExpr>(expr)) {
    auto value = generateExpr(temporary->getSubExpr());
    if (!value) {
      return mlir::failure();
    }
    if (expr->getType()->isArrayType()) {
      if (auto memref = mlir::dyn_cast<mlir::MemRefType>(value.getType())) {
        return clang_mlir::Address::get(value, memref.getElementType());
      }
    }
    if (expr->getType()->isPointerType() &&
        mlir::isa<mlir::MemRefType>(value.getType())) {
      value = clang_mlir::toValue(builder, loc, value,
                                  types.convertType(expr->getType()));
      if (!value) {
        return mlir::failure();
      }
    }
    auto storage = clang_mlir::createLocalSlot(builder, loc, value);
    return clang_mlir::Address::get(storage, value.getType());
  }
  if (auto *array = llvm::dyn_cast<clang::ArraySubscriptExpr>(expr)) {
    return emitArrayAddress(array);
  }
  mlir::Value base;
  llvm::SmallVector<mlir::Value, 4> indices;
  if (auto *call = llvm::dyn_cast<clang::CallExpr>(expr)) {
    auto result = generateCallExpr(call);
    if (mlir::failed(result)) {
      return mlir::failure();
    }
    base = *result;
  } else if (auto *literal = llvm::dyn_cast<clang::StringLiteral>(expr)) {
    base = generateStringLiteral(literal);
  } else if (auto *decl = llvm::dyn_cast<clang::DeclRefExpr>(expr)) {
    if (auto *var = llvm::dyn_cast<clang::VarDecl>(decl->getDecl())) {
      auto it = function.locals.find(var);
      if (it != function.locals.end()) {
        base = it->second;
      } else {
        context.source().error(expr->getExprLoc(),
                               "object has no addressable storage");
        return mlir::failure();
      }
    }
  } else if (auto *member = llvm::dyn_cast<clang::MemberExpr>(expr)) {
    return emitMemberAddress(member);
  } else if (auto *unary = llvm::dyn_cast<clang::UnaryOperator>(expr)) {
    if (unary->getOpcode() == clang::UO_Deref) {
      base = generateExpr(unary->getSubExpr());
      if (base) {
        if (auto memref = mlir::dyn_cast<mlir::MemRefType>(base.getType())) {
          if (memref.getRank() > 0) {
            indices.push_back(clang_mlir::indexConst(builder, loc, 0));
          }
        }
      }
    }
  }
  mlir::Type element = types.convertStorageType(expr->getType());
  if (base) {
    if (auto memref = mlir::dyn_cast<mlir::MemRefType>(base.getType())) {
      element = memref.getElementType();
    }
  }
  auto address = clang_mlir::Address::get(base, element, indices);
  if (mlir::failed(address)) {
    context.source().error(expr->getExprLoc(),
                           "unsupported or invalid object address");
  }
  return address;
}

auto ExpressionLowering::addressOf(clang::Expr *expr)
    -> mlir::FailureOr<clang_mlir::Address> {
  auto &builder = context.builder();
  auto loc = context.source().location(expr->getExprLoc());
  auto target =
      types.convertType(context.ast().getLValueReferenceType(expr->getType()));
  if (!target) {
    return mlir::failure();
  }
  auto emitted = emitLValue(
      expr, target,
      [&](const LValueResult &result) -> mlir::FailureOr<mlir::Value> {
        const auto *object = std::get_if<ObjectResult>(&result);
        if (!object) {
          context.source().error(expr->getExprLoc(),
                                 "object has no addressable storage");
          return mlir::failure();
        }
        return object->address.materialize(builder, loc);
      });
  if (mlir::failed(emitted)) {
    return mlir::failure();
  }
  auto base = *emitted;
  auto memref = base ? mlir::dyn_cast<mlir::MemRefType>(base.getType())
                     : mlir::MemRefType{};
  return clang_mlir::Address::get(
      base, memref ? memref.getElementType()
                   : types.convertStorageType(expr->getType()));
}

auto ExpressionLowering::loadLValue(clang::Expr *expr) -> mlir::Value {
  auto &builder = context.builder();
  auto loc = context.source().location(expr->getExprLoc());
  auto emitted = emitLValue(
      expr, types.convertType(expr->getType()),
      [&](const LValueResult &result) -> mlir::FailureOr<mlir::Value> {
        if (const auto *value = std::get_if<mlir::Value>(&result)) {
          return *value;
        }
        const auto &object = std::get<ObjectResult>(result);
        auto value = object.computed ? object.computed
                                     : object.address.load(builder, loc);
        if (!value) {
          return mlir::failure();
        }
        return value;
      });
  return mlir::succeeded(emitted) ? *emitted : mlir::Value{};
}

} // namespace cmlir::frontend
