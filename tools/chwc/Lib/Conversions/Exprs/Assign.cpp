#include "../../Converter.h"
#include "llvm/Support/WithColor.h"

namespace chwc {

auto ignoreCasts(clang::Expr *expr) -> clang::Expr * {
  return expr ? expr->IgnoreParenImpCasts() : nullptr;
}

auto CHWConverter::generateAssignmentBinaryOperator(
    clang::BinaryOperator *assignOp) -> mlir::Value {
  clang::Expr *lhs = assignOp->getLHS();
  clang::Expr *rhs = assignOp->getRHS();

  mlir::Value rhsValue = generateExpr(rhs);
  if (!rhsValue) {
    return nullptr;
  }

  const clang::FieldDecl *fieldDecl = getAssignedField(lhs);

  if (fieldDecl) {
    auto fieldIt = fieldTable.find(fieldDecl);
    if (fieldIt == fieldTable.end()) {
      llvm::WithColor::error()
          << "chwc: assignment lhs is not hardware field\n";
      return rhsValue;
    }

    HWFieldInfo &fieldInfo = fieldIt->second;

    switch (fieldInfo.kind) {
    case HWFieldKind::Input:
      llvm::WithColor::error() << "chwc: cannot assign to hardware input\n";
      break;

    case HWFieldKind::Output:
      outputValueTable[fieldDecl] = rhsValue;
      break;

    case HWFieldKind::Reg:
      nextFieldValueTable[fieldDecl] = rhsValue;
      break;

    case HWFieldKind::Wire:
      currentFieldValueTable[fieldDecl] = rhsValue;
      break;
    }

    return rhsValue;
  }

  auto *declRef = mlir::dyn_cast_or_null<clang::DeclRefExpr>(ignoreCasts(lhs));
  if (declRef) {
    if (auto *varDecl = mlir::dyn_cast<clang::VarDecl>(declRef->getDecl())) {
      localValueTable[varDecl] = rhsValue;
      return rhsValue;
    }
  }

  llvm::WithColor::error() << "chwc: unsupported assignment lhs\n";
  return rhsValue;
}

} // namespace chwc
