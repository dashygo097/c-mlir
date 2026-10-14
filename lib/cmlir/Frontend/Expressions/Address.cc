#include "../Lowering.h"
#include "clang-mlir/Constants.h"
#include "clang-mlir/Numerics.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

namespace cmlir::frontend {

auto ASTLowering::emitAddress(clang::Expr *expr)
    -> mlir::FailureOr<clang_mlir::Address> {
  expr = expr->IgnoreParens();
  auto &builder = context.builder();
  auto loc = context.source().location(expr->getExprLoc());
  if (auto *cast = llvm::dyn_cast<clang::ImplicitCastExpr>(expr)) {
    if (cast->getCastKind() == clang::CK_NoOp) {
      return emitAddress(cast->getSubExpr());
    }
  }
  if (auto *binary = llvm::dyn_cast<clang::BinaryOperator>(expr)) {
    if (binary->isAssignmentOp()) {
      return emitAssignmentAddress(binary);
    }
  }
  if (auto *array = llvm::dyn_cast<clang::ArraySubscriptExpr>(expr)) {
    return emitArrayAddress(array);
  }
  mlir::Value base;
  llvm::SmallVector<mlir::Value, 4> indices;
  if (auto *decl = llvm::dyn_cast<clang::DeclRefExpr>(expr)) {
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
    base = generateMemberExpr(member);
  } else if (auto *unary = llvm::dyn_cast<clang::UnaryOperator>(expr)) {
    if (unary->isIncrementDecrementOp() && unary->isPrefix()) {
      auto address = emitAddress(unary->getSubExpr());
      if (mlir::failed(address)) {
        return mlir::failure();
      }
      auto old = address->load(builder, loc);
      auto value = unary->isIncrementOp() ? clang_mlir::inc(builder, loc, old)
                                          : clang_mlir::dec(builder, loc, old);
      if (!value || mlir::failed(address->store(builder, loc, value))) {
        return mlir::failure();
      }
      return address;
    }
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
  mlir::Type element = convertType(expr->getType());
  if (base && mlir::isa<mlir::LLVM::LLVMPointerType>(base.getType()) &&
      expr->getType()->isPointerType()) {
    element = mlir::LLVM::LLVMPointerType::get(builder.getContext());
  }
  auto address = clang_mlir::Address::get(base, element, indices);
  if (mlir::failed(address)) {
    context.source().error(expr->getExprLoc(),
                           "unsupported or invalid object address");
  }
  return address;
}

auto ASTLowering::loadLValue(clang::Expr *expr) -> mlir::Value {
  expr = expr->IgnoreParens();
  if (auto *conditional = llvm::dyn_cast<clang::ConditionalOperator>(expr)) {
    return generateConditionalOperator(conditional);
  }
  if (auto *ref = llvm::dyn_cast<clang::DeclRefExpr>(expr)) {
    if (auto *param = llvm::dyn_cast<clang::ParmVarDecl>(ref->getDecl())) {
      if (function.parameters.count(param)) {
        return function.parameters[param];
      }
    }
  }
  if (auto *member = llvm::dyn_cast<clang::MemberExpr>(expr)) {
    auto *base = member->getBase()->IgnoreParenImpCasts();
    while (auto *parent = llvm::dyn_cast<clang::MemberExpr>(base)) {
      base = parent->getBase()->IgnoreParenImpCasts();
    }
    if (auto *ref = llvm::dyn_cast<clang::DeclRefExpr>(base)) {
      if (auto *param = llvm::dyn_cast<clang::ParmVarDecl>(ref->getDecl())) {
        if (function.parameters.count(param) &&
            mlir::isa<mlir::LLVM::LLVMStructType>(
                function.parameters[param].getType())) {
          return generateMemberExpr(member);
        }
      }
    }
  }
  auto address = emitAddress(expr);
  return mlir::succeeded(address)
             ? address->load(context.builder(),
                             context.source().location(expr->getExprLoc()))
             : mlir::Value{};
}

} // namespace cmlir::frontend
