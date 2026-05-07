#include "../../Converter.h"
#include "../Utils/Expr.h"
#include "../Utils/Type.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/OperationKinds.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/WithColor.h"

namespace chwc {

auto CHWConverter::generateArraySubscriptExpr(
    clang::ArraySubscriptExpr *arraySub) -> mlir::Value {
  if (!arraySub) {
    return nullptr;
  }

  const clang::FieldDecl *fieldDecl =
      utils::getArrayBaseFieldDecl(arraySub->getBase());
  if (!fieldDecl) {
    llvm::WithColor::error()
        << "chwc: array read only supports hardware field arrays\n";
    return nullptr;
  }

  auto lowerIndex = [&](auto &&self, clang::Expr *expr) -> mlir::Value {
    if (!expr) {
      return nullptr;
    }

    expr = utils::ignoreExprWrappers(expr);

    if (auto *implicitCast =
            llvm::dyn_cast_or_null<clang::ImplicitCastExpr>(expr)) {
      using CK = clang::CastKind;

      switch (implicitCast->getCastKind()) {
      case CK::CK_LValueToRValue:
      case CK::CK_NoOp:
      case CK::CK_IntegralCast:
      case CK::CK_UserDefinedConversion:
      case CK::CK_ConstructorConversion:
        return self(self, implicitCast->getSubExpr());

      default:
        break;
      }
    }

    if (auto *explicitCast =
            llvm::dyn_cast_or_null<clang::ExplicitCastExpr>(expr)) {
      using CK = clang::CastKind;

      switch (explicitCast->getCastKind()) {
      case CK::CK_LValueToRValue:
      case CK::CK_NoOp:
      case CK::CK_IntegralCast:
      case CK::CK_UserDefinedConversion:
      case CK::CK_ConstructorConversion:
        return self(self, explicitCast->getSubExpr());

      default:
        break;
      }
    }

    if (auto *memberCall =
            llvm::dyn_cast_or_null<clang::CXXMemberCallExpr>(expr)) {
      auto *conversionDecl = llvm::dyn_cast_or_null<clang::CXXConversionDecl>(
          memberCall->getMethodDecl());

      if (conversionDecl) {
        clang::Expr *objectExpr = memberCall->getImplicitObjectArgument();
        if (objectExpr &&
            utils::getSignalTypeInfo(objectExpr->getType()).isValue) {
          return self(self, objectExpr);
        }
      }
    }

    return generateExpr(expr);
  };

  mlir::Value index = lowerIndex(lowerIndex, arraySub->getIdx());
  if (!index) {
    llvm::WithColor::error() << "chwc: failed to generate array index\n";
    return nullptr;
  }

  return readArrayElement(fieldDecl, index);
}

} // namespace chwc
