#include "../../Converter.h"
#include "clang/Basic/SourceManager.h"

namespace cmlirc {
bool CMLIRConverter::TraverseFunctionDecl(clang::FunctionDecl *decl) {
  if (decl->isImplicit() || !decl->hasBody()) {
    return true;
  }

  // skip system headers
  clang::SourceManager &sm = contextManager.ClangContext().getSourceManager();
  if (sm.isInSystemHeader(decl->getLocation())) {
    return true;
  }

  mlir::OpBuilder &builder = contextManager.Builder();
  builder.setInsertionPointToEnd(contextManager.Module().getBody());

  // Convert parameter types
  llvm::SmallVector<mlir::Type, 4> argTypes;
  for (auto *param : decl->parameters()) {
    mlir::Type paramType = convertType(param->getType());
    argTypes.push_back(paramType);
  }

  // Convert return type
  mlir::Type returnType = convertType(decl->getReturnType());

  llvm::SmallVector<mlir::Type, 1> returnTypes;
  if (!mlir::isa<mlir::NoneType>(returnType)) {
    returnTypes.push_back(returnType);
  }

  auto funcType = builder.getFunctionType(argTypes, returnTypes);

  // Create function
  auto funcOp = mlir::func::FuncOp::create(builder, builder.getUnknownLoc(),
                                           decl->getNameAsString(), funcType);

  // Create entry block
  auto *entryBlock = funcOp.addEntryBlock();
  builder.setInsertionPointToStart(entryBlock);

  currentFunc = funcOp;

  // Map parameters to block arguments
  for (uint32_t i = 0; i < decl->getNumParams(); ++i) {
    auto *param = decl->getParamDecl(i);
    paramTable[param] = entryBlock->getArgument(i);
  }

  // Traverse function body manually
  TraverseStmt(decl->getBody());

  // Ensure the function has a terminator
  builder.setInsertionPointToEnd(entryBlock);
  if (entryBlock->empty() ||
      !entryBlock->back().hasTrait<mlir::OpTrait::IsTerminator>()) {
    mlir::func::ReturnOp::create(builder, builder.getUnknownLoc());
  }

  return true;
}

} // namespace cmlirc
