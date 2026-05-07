#include "../../Converter.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/Support/WithColor.h"

namespace chwc {

auto evaluateNonTypeTemplateParamDefault(
    clang::NonTypeTemplateParmDecl *paramDecl, mlir::OpBuilder &builder,
    mlir::Type type) -> mlir::Attribute {
  if (!paramDecl || !paramDecl->hasDefaultArgument()) {
    return {};
  }

  const clang::TemplateArgumentLoc &defaultArgLoc =
      paramDecl->getDefaultArgument();
  const clang::TemplateArgument &defaultArg = defaultArgLoc.getArgument();

  if (defaultArg.getKind() == clang::TemplateArgument::Integral) {
    llvm::APSInt value = defaultArg.getAsIntegral();
    return builder.getIntegerAttr(type, value);
  }

  clang::Expr *defaultExpr = defaultArgLoc.getSourceExpression();
  if (!defaultExpr) {
    llvm::WithColor::error()
        << "chwc: template parameter default has no source expression: "
        << paramDecl->getNameAsString() << "\n";
    return {};
  }

  clang::Expr::EvalResult result;
  clang::ASTContext &astContext = paramDecl->getASTContext();

  if (!defaultExpr->EvaluateAsInt(result, astContext)) {
    llvm::WithColor::error()
        << "chwc: template parameter default must be integer constant: "
        << paramDecl->getNameAsString() << "\n";
    return {};
  }

  llvm::APSInt value = result.Val.getInt();
  return builder.getIntegerAttr(type, value);
}

auto CHWConverter::TraverseNonTypeTemplateParmDecl(
    clang::NonTypeTemplateParmDecl *paramDecl) -> bool {
  if (!paramDecl) {
    return true;
  }

  if (!paramDecl->getType()->isIntegralOrEnumerationType()) {
    llvm::WithColor::error()
        << "chwc: template parameter must be integral or enum: "
        << paramDecl->getNameAsString() << "\n";
    return true;
  }

  mlir::Type type = convertType(paramDecl->getType());
  if (!type) {
    llvm::WithColor::error() << "chwc: unsupported template parameter type: "
                             << paramDecl->getType().getAsString() << "\n";
    return true;
  }

  mlir::OpBuilder &builder = contextManager.Builder();

  HWParamInfo paramInfo;
  paramInfo.paramDecl = paramDecl;
  paramInfo.name = paramDecl->getNameAsString();
  paramInfo.type = type;
  paramInfo.defaultValue =
      evaluateNonTypeTemplateParamDefault(paramDecl, builder, type);

  hardwareParamOrder.push_back(paramDecl);
  paramTable[paramDecl] = paramInfo;

  return true;
}

} // namespace chwc
