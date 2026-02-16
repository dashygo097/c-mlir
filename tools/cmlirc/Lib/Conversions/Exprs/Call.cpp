#include "../../Converter.h"
#include "../Types/Types.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace cmlirc {

mlir::func::FuncOp getOrCreateFunctionDecl(mlir::OpBuilder &builder,
                                           mlir::ModuleOp module,
                                           const std::string &name,
                                           mlir::FunctionType funcType) {
  if (auto existing = module.lookupSymbol<mlir::func::FuncOp>(name)) {
    return existing;
  }

  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(module.getBody());

  auto funcOp = mlir::func::FuncOp::create(builder, builder.getUnknownLoc(),
                                           name, funcType);
  funcOp.setPrivate();

  return funcOp;
}

mlir::Value CMLIRConverter::generateCallExpr(clang::CallExpr *callExpr) {
  mlir::OpBuilder &builder = context_manager_.Builder();
  mlir::Location loc = builder.getUnknownLoc();

  const clang::FunctionDecl *calleeDecl = callExpr->getDirectCallee();
  if (!calleeDecl) {
    llvm::errs() << "Indirect calls not supported yet\n";
    return nullptr;
  }

  std::string calleeName = calleeDecl->getNameAsString();

  llvm::SmallVector<mlir::Value, 4> argValues;
  llvm::SmallVector<mlir::Type, 4> argTypes;

  for (unsigned i = 0; i < callExpr->getNumArgs(); ++i) {
    clang::Expr *argExpr = callExpr->getArg(i);
    mlir::Value argValue = generateExpr(argExpr);
    if (!argValue) {
      llvm::errs() << "Failed to generate argument " << i << "\n";
      return nullptr;
    }
    argValues.push_back(argValue);
    argTypes.push_back(argValue.getType());
  }

  clang::QualType returnType = calleeDecl->getReturnType();
  mlir::Type mlirReturnType = convertType(builder, returnType);

  llvm::SmallVector<mlir::Type, 1> returnTypes;
  if (!mlir::isa<mlir::NoneType>(mlirReturnType)) {
    returnTypes.push_back(mlirReturnType);
  }

  auto funcType = builder.getFunctionType(argTypes, returnTypes);

  mlir::ModuleOp module = context_manager_.Module();
  getOrCreateFunctionDecl(builder, module, calleeName, funcType);

  auto callOp = mlir::func::CallOp::create(builder, loc, calleeName,
                                           returnTypes, argValues);

  if (callOp.getNumResults() > 0) {
    return callOp.getResult(0);
  }

  return nullptr;
}

} // namespace cmlirc
