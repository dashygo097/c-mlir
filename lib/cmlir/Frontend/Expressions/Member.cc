#include "../Lowering.h"
#include "clang-mlir/Constants.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "clang/AST/Expr.h"
#include "llvm/Support/WithColor.h"

namespace cmlir::frontend {

auto ASTLowering::generateMemberExpr(clang::MemberExpr *memberExpr)
    -> mlir::Value {
  mlir::OpBuilder &builder = context.builder();
  mlir::Location loc = context.source().location(memberExpr->getExprLoc());

  clang::Expr *base = memberExpr->getBase();
  mlir::Value baseValue = generateExpr(base);
  if (!baseValue) {
    return nullptr;
  }

  auto *fieldDecl =
      mlir::dyn_cast<clang::FieldDecl>(memberExpr->getMemberDecl());
  if (!fieldDecl) {
    llvm::WithColor::error() << "clang-to-mlir: Member is not a field\n";
    return nullptr;
  }

  clang::QualType baseType = base->getType().getCanonicalType();

  if (memberExpr->isArrow()) {
    if (!baseType->isPointerType()) {
      llvm::WithColor::error()
          << "clang-to-mlir: arrow member base is not pointer type\n";
      return nullptr;
    }

    baseType = baseType->getPointeeType().getCanonicalType();
  }

  const auto *recordType = baseType->getAs<clang::RecordType>();
  if (!recordType) {
    llvm::WithColor::error()
        << "clang-to-mlir: Base is not a struct or class type\n";
    return nullptr;
  }

  const clang::RecordDecl *recordDecl = recordType->getDecl();
  const clang::RecordDecl *definition = recordDecl->getDefinition();
  if (!definition) {
    definition = recordDecl;
  }

  std::optional<uint32_t> fieldIndexOpt =
      types.getFieldIndex(definition, fieldDecl);
  if (!fieldIndexOpt) {
    llvm::WithColor::error()
        << "clang-to-mlir: field not found in struct/class\n";
    return nullptr;
  }

  uint32_t fieldIndex = *fieldIndexOpt;

  mlir::Type structType = convertType(baseType);
  auto llvmStructType =
      mlir::dyn_cast_or_null<mlir::LLVM::LLVMStructType>(structType);
  if (!llvmStructType) {
    llvm::WithColor::error() << "clang-to-mlir: expected LLVM struct type\n";
    return nullptr;
  }

  if (mlir::isa<mlir::LLVM::LLVMPointerType>(baseValue.getType())) {
    auto ptrType = mlir::LLVM::LLVMPointerType::get(builder.getContext());

    llvm::SmallVector<mlir::Value, 2> indices;
    indices.push_back(
        clang_mlir::intConst(builder, loc, builder.getI32Type(), 0));
    indices.push_back(
        clang_mlir::intConst(builder, loc, builder.getI32Type(), fieldIndex));

    auto fieldPtr = mlir::LLVM::GEPOp::create(
        builder, loc, ptrType, llvmStructType, baseValue, indices);

    return fieldPtr.getResult();
  }

  if (mlir::isa<mlir::LLVM::LLVMStructType>(baseValue.getType())) {
    auto fieldValue =
        mlir::LLVM::ExtractValueOp::create(builder, loc, baseValue, fieldIndex);
    return fieldValue.getResult();
  }

  llvm::WithColor::error()
      << "clang-to-mlir: unsupported base type for member access\n";
  return nullptr;
}

} // namespace cmlir::frontend
