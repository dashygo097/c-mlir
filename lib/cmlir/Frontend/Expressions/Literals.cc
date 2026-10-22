#include "../Lowering.h"
#include "clang-mlir/Constants.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

namespace cmlir::frontend {

auto ASTLowering::generateCXXBoolLiteralExpr(clang::CXXBoolLiteralExpr *boolLit)
    -> mlir::Value {
  mlir::OpBuilder &builder = context.builder();
  mlir::Location loc = builder.getUnknownLoc();

  bool value = boolLit->getValue();

  return clang_mlir::boolConst(builder, loc, value);
}

auto ASTLowering::generateIntegerLiteral(clang::IntegerLiteral *intLit)
    -> mlir::Value {
  mlir::OpBuilder &builder = context.builder();
  mlir::Location loc = builder.getUnknownLoc();

  auto value = intLit->getValue();
  mlir::Type type = types.convertType(intLit->getType());

  return clang_mlir::intConst(builder, loc, type, value);
}

auto ASTLowering::generateFloatingLiteral(clang::FloatingLiteral *floatLit)
    -> mlir::Value {
  mlir::OpBuilder &builder = context.builder();
  mlir::Location loc = builder.getUnknownLoc();

  auto value = floatLit->getValue();
  mlir::Type type = types.convertType(floatLit->getType());

  return clang_mlir::floatConst(builder, loc, type, value);
}

auto ASTLowering::generateCharacterLiteral(clang::CharacterLiteral *charLit)
    -> mlir::Value {
  mlir::OpBuilder &builder = context.builder();
  mlir::Location loc = builder.getUnknownLoc();

  uint64_t value = charLit->getValue();
  mlir::Type type = types.convertType(charLit->getType());

  return clang_mlir::intConst(builder, loc, type, value);
}

auto ASTLowering::generateStringLiteral(clang::StringLiteral *strLit)
    -> mlir::Value {
  auto &builder = context.builder();
  auto loc = builder.getUnknownLoc();

  llvm::StringRef str = strLit->getString();
  auto nullTerminated = (str + llvm::StringRef("\0", 1)).str();
  auto strAttr = builder.getStringAttr(nullTerminated);

  auto arrayType = mlir::LLVM::LLVMArrayType::get(builder.getI8Type(),
                                                  nullTerminated.size());

  auto modulOp =
      builder.getBlock()->getParent()->getParentOfType<mlir::ModuleOp>();
  std::string symName =
      "__str_" +
      std::to_string(std::hash<std::string>{}(nullTerminated.c_str()));

  if (!modulOp.lookupSymbol(symName)) {
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(modulOp.getBody());
    mlir::LLVM::GlobalOp::create(
        builder, loc, arrayType,
        /*isConstant=*/true, mlir::LLVM::Linkage::Internal, symName, strAttr);
  }

  auto ptrType = mlir::LLVM::LLVMPointerType::get(builder.getContext());
  return mlir::LLVM::AddressOfOp::create(builder, loc, ptrType, symName)
      .getResult();
}

} // namespace cmlir::frontend
