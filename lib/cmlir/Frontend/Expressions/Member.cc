#include "../Lowering.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

namespace cmlir::frontend {

auto ASTLowering::emitMemberAddress(clang::MemberExpr *expr)
    -> mlir::FailureOr<clang_mlir::Address> {
  auto &builder = context.builder();
  auto loc = context.source().location(expr->getExprLoc());
  auto *base = expr->getBase();
  mlir::Value pointer;
  auto type = base->getType();
  if (expr->isArrow()) {
    type = type->getPointeeType();
    pointer = generateExpr(base);
  } else {
    auto address = emitAddress(base);
    if (mlir::failed(address)) {
      return mlir::failure();
    }
    pointer = address->materialize(builder, loc);
  }
  auto *field = llvm::dyn_cast<clang::FieldDecl>(expr->getMemberDecl());
  auto layout = types.convertStorageType(type);
  auto index = field ? types.getFieldIndex(type->getAsRecordDecl(), field)
                     : std::nullopt;
  if (!pointer || !layout || !index ||
      !llvm::isa<mlir::LLVM::LLVMPointerType>(pointer.getType())) {
    context.source().error(expr->getExprLoc(), "invalid member address");
    return mlir::failure();
  }
  auto fieldPointer = mlir::LLVM::GEPOp::create(
      builder, loc, pointer.getType(), layout, pointer,
      llvm::ArrayRef<mlir::LLVM::GEPArg>{0, static_cast<int32_t>(*index)});
  return clang_mlir::Address::get(fieldPointer,
                                  types.convertStorageType(field->getType()));
}

auto ASTLowering::generateMemberExpr(clang::MemberExpr *expr) -> mlir::Value {
  auto &builder = context.builder();
  auto loc = context.source().location(expr->getExprLoc());
  if (expr->isGLValue() || expr->isArrow() || expr->getType()->isArrayType()) {
    auto address = emitMemberAddress(expr);
    if (mlir::failed(address)) {
      return {};
    }
    return expr->getType()->isArrayType() ? address->materialize(builder, loc)
                                          : address->load(builder, loc);
  }
  auto *base = expr->getBase();
  auto value = generateExpr(base);
  auto *field = llvm::dyn_cast<clang::FieldDecl>(expr->getMemberDecl());
  auto index =
      field ? types.getFieldIndex(base->getType()->getAsRecordDecl(), field)
            : std::nullopt;
  if (!value || !index ||
      !llvm::isa<mlir::LLVM::LLVMStructType>(value.getType())) {
    context.source().error(expr->getExprLoc(), "invalid member value");
    return {};
  }
  return mlir::LLVM::ExtractValueOp::create(builder, loc, value, *index);
}

} // namespace cmlir::frontend
