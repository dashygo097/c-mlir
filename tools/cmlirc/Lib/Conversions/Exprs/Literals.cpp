#include "../../Converter.h"
#include "../Utils/Constants.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

namespace cmlirc {

mlir::Value
CMLIRConverter::generateCXXBoolLiteralExpr(clang::CXXBoolLiteralExpr *boolLit) {
  mlir::OpBuilder &builder = context_manager_.Builder();
  mlir::Location loc = builder.getUnknownLoc();

  bool value = boolLit->getValue();

  return detail::boolConst(builder, loc, value);
}

mlir::Value
CMLIRConverter::generateIntegerLiteral(clang::IntegerLiteral *intLit) {
  mlir::OpBuilder &builder = context_manager_.Builder();
  mlir::Location loc = builder.getUnknownLoc();

  int64_t value = intLit->getValue().getSExtValue();
  mlir::Type type = convertType(intLit->getType());

  return detail::intConst(builder, loc, type, value);
}

mlir::Value
CMLIRConverter::generateFloatingLiteral(clang::FloatingLiteral *floatLit) {
  mlir::OpBuilder &builder = context_manager_.Builder();
  mlir::Location loc = builder.getUnknownLoc();

  auto value = floatLit->getValue();
  mlir::Type type = convertType(floatLit->getType());

  return detail::floatConst(builder, loc, type, value.convertToDouble());
}

mlir::Value
CMLIRConverter::generateCharacterLiteral(clang::CharacterLiteral *charLit) {
  mlir::OpBuilder &builder = context_manager_.Builder();
  mlir::Location loc = builder.getUnknownLoc();

  uint64_t value = charLit->getValue();
  mlir::Type type = convertType(charLit->getType());

  return detail::intConst(builder, loc, type, value);
}

mlir::Value
CMLIRConverter::generateStringLiteral(clang::StringLiteral *strLit) {
  auto &builder = context_manager_.Builder();
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

} // namespace cmlirc
