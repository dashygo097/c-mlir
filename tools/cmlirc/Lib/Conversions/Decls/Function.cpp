#include "../../Converter.h"
#include "clang/Basic/SourceManager.h"

namespace cmlirc {

auto CMLIRConverter::TraverseFunctionDecl(clang::FunctionDecl *decl) -> bool {
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

  // Name of the function
  std::string funcName = decl->getNameAsString();

  // Convert parameter types
  llvm::SmallVector<mlir::Type, 4> argTypes;

  // Check MethodDecl
  bool isInstanceMethod = false;
  if (auto *methodDecl = mlir::dyn_cast<clang::CXXMethodDecl>(decl)) {
    if (!methodDecl->isStatic()) {
      isInstanceMethod = true;
      mlir::Type thisType = convertType(methodDecl->getThisType());
      funcName =
          "__" + methodDecl->getParent()->getNameAsString() + "_" + funcName;
      argTypes.push_back(thisType);
    }
  }

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
                                           funcName, funcType);

  // Create entry block
  auto *entryBlock = funcOp.addEntryBlock();
  builder.setInsertionPointToStart(entryBlock);

  currentFunc = funcOp;

  uint32_t argIdx = 0;
  if (isInstanceMethod) {
    currentThisValue = entryBlock->getArgument(argIdx++);
  } else {
    currentThisValue = nullptr;
  }

  // Map parameters to block arguments
  for (uint32_t i = 0; i < decl->getNumParams(); ++i) {
    auto *param = decl->getParamDecl(i);
    paramTable[param] = entryBlock->getArgument(argIdx++);
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
