#include "../../Converter.h"
#include "../Utils/Cast.h"
#include "../Utils/Comb.h"
#include "../Utils/Expr.h"
#include "../Utils/Type.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/OperationKinds.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/WithColor.h"

namespace chwc {

auto isOverloadedCompoundAssign(clang::OverloadedOperatorKind op) -> bool {
  using OO = clang::OverloadedOperatorKind;

  switch (op) {
  case OO::OO_PlusEqual:
  case OO::OO_MinusEqual:
  case OO::OO_StarEqual:
  case OO::OO_SlashEqual:
  case OO::OO_PercentEqual:
  case OO::OO_AmpEqual:
  case OO::OO_PipeEqual:
  case OO::OO_CaretEqual:
  case OO::OO_LessLessEqual:
  case OO::OO_GreaterGreaterEqual:
    return true;
  default:
    return false;
  }
}

auto emitOverloadedCompoundArith(mlir::OpBuilder &builder, mlir::Location loc,
                                 clang::OverloadedOperatorKind op,
                                 mlir::Value lhs, mlir::Value rhs,
                                 bool isSigned) -> mlir::Value {
  using OO = clang::OverloadedOperatorKind;

  switch (op) {
  case OO::OO_PlusEqual:
    return utils::add(builder, loc, lhs, rhs);

  case OO::OO_MinusEqual:
    return utils::sub(builder, loc, lhs, rhs);

  case OO::OO_StarEqual:
    return utils::mul(builder, loc, lhs, rhs);

  case OO::OO_SlashEqual:
    return isSigned ? utils::divS(builder, loc, lhs, rhs)
                    : utils::divU(builder, loc, lhs, rhs);

  case OO::OO_PercentEqual:
    return isSigned ? utils::modS(builder, loc, lhs, rhs)
                    : utils::modU(builder, loc, lhs, rhs);

  case OO::OO_AmpEqual:
    return utils::bitAnd(builder, loc, lhs, rhs);

  case OO::OO_PipeEqual:
    return utils::bitOr(builder, loc, lhs, rhs);

  case OO::OO_CaretEqual:
    return utils::bitXor(builder, loc, lhs, rhs);

  case OO::OO_LessLessEqual:
    return utils::shl(builder, loc, lhs, rhs);

  case OO::OO_GreaterGreaterEqual:
    return isSigned ? utils::shrS(builder, loc, lhs, rhs)
                    : utils::shrU(builder, loc, lhs, rhs);

  default:
    llvm::WithColor::error()
        << "chwc: unsupported overloaded compound assignment\n";
    return nullptr;
  }
}

auto getOperatorLHS(clang::CXXOperatorCallExpr *callExpr) -> clang::Expr * {
  if (!callExpr || callExpr->getNumArgs() < 1) {
    return nullptr;
  }

  return callExpr->getArg(0);
}

auto getOperatorRHS(clang::CXXOperatorCallExpr *callExpr) -> clang::Expr * {
  if (!callExpr || callExpr->getNumArgs() < 2) {
    return nullptr;
  }

  return callExpr->getArg(1);
}

auto CHWConverter::generateCXXOperatorCallExpr(
    clang::CXXOperatorCallExpr *callExpr) -> mlir::Value {
  mlir::OpBuilder &builder = contextManager.Builder();
  mlir::Location loc = builder.getUnknownLoc();

  using OO = clang::OverloadedOperatorKind;

  OO op = callExpr->getOperator();

  auto lowerArrayIndex = [&](auto &&self, clang::Expr *expr) -> mlir::Value {
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

  if (op == OO::OO_Equal) {
    clang::Expr *lhsExpr = getOperatorLHS(callExpr);
    clang::Expr *rhsExpr = getOperatorRHS(callExpr);

    if (!lhsExpr || !rhsExpr) {
      llvm::WithColor::error()
          << "chwc: malformed overloaded assignment operator\n";
      return nullptr;
    }

    mlir::Value rhsValue = generateExpr(rhsExpr);
    if (!rhsValue) {
      llvm::WithColor::error()
          << "chwc: failed to generate RHS for overloaded assignment\n";
      return nullptr;
    }

    if (auto *arraySub = llvm::dyn_cast_or_null<clang::ArraySubscriptExpr>(
            utils::ignoreExprWrappers(lhsExpr))) {
      const clang::FieldDecl *fieldDecl =
          utils::getArrayBaseFieldDecl(arraySub->getBase());
      if (!fieldDecl) {
        llvm::WithColor::error()
            << "chwc: array assignment only supports hardware field arrays\n";
        return rhsValue;
      }

      mlir::Value index = lowerArrayIndex(lowerArrayIndex, arraySub->getIdx());
      if (!index) {
        llvm::WithColor::error()
            << "chwc: failed to generate array assignment index\n";
        return rhsValue;
      }

      return assignArrayElement(fieldDecl, index, rhsValue);
    }

    const clang::FieldDecl *fieldDecl = getAssignedField(lhsExpr);
    if (fieldDecl) {
      return assignFieldValue(fieldDecl, rhsValue);
    }

    auto *declRef = llvm::dyn_cast_or_null<clang::DeclRefExpr>(
        utils::ignoreExprWrappers(lhsExpr));
    if (declRef) {
      if (auto *varDecl = llvm::dyn_cast<clang::VarDecl>(declRef->getDecl())) {
        mlir::Type targetType = convertType(varDecl->getType());
        if (targetType) {
          rhsValue = utils::promoteValue(builder, loc, rhsValue, targetType);
        }

        localValueTable[varDecl] = rhsValue;
        return rhsValue;
      }
    }

    llvm::WithColor::error() << "chwc: unsupported overloaded assignment lhs\n";
    return rhsValue;
  }

  if (isOverloadedCompoundAssign(op)) {
    clang::Expr *lhsExpr = getOperatorLHS(callExpr);
    clang::Expr *rhsExpr = getOperatorRHS(callExpr);

    if (!lhsExpr || !rhsExpr) {
      llvm::WithColor::error()
          << "chwc: malformed overloaded compound assignment operator\n";
      return nullptr;
    }

    mlir::Value oldValue = generateExpr(lhsExpr);
    mlir::Value rhsValue = generateExpr(rhsExpr);

    if (!oldValue || !rhsValue) {
      llvm::WithColor::error() << "chwc: failed to generate overloaded "
                                  "compound assignment operands\n";
      return nullptr;
    }

    rhsValue = utils::promoteValue(builder, loc, rhsValue, oldValue.getType());
    if (!rhsValue) {
      return nullptr;
    }

    clang::QualType objectType =
        lhsExpr ? lhsExpr->getType() : clang::QualType{};
    auto typeInfo = utils::getSignalTypeInfo(objectType);

    mlir::Value resultValue = emitOverloadedCompoundArith(
        builder, loc, op, oldValue, rhsValue, typeInfo.isSigned);

    if (!resultValue) {
      return nullptr;
    }

    if (auto *arraySub = llvm::dyn_cast_or_null<clang::ArraySubscriptExpr>(
            utils::ignoreExprWrappers(lhsExpr))) {
      const clang::FieldDecl *fieldDecl =
          utils::getArrayBaseFieldDecl(arraySub->getBase());
      if (!fieldDecl) {
        llvm::WithColor::error()
            << "chwc: array compound assignment only supports hardware field "
               "arrays\n";
        return resultValue;
      }

      mlir::Value index = lowerArrayIndex(lowerArrayIndex, arraySub->getIdx());
      if (!index) {
        llvm::WithColor::error()
            << "chwc: failed to generate array compound assignment index\n";
        return resultValue;
      }

      return assignArrayElement(fieldDecl, index, resultValue);
    }

    const clang::FieldDecl *fieldDecl = getAssignedField(lhsExpr);
    if (fieldDecl) {
      return assignFieldValue(fieldDecl, resultValue);
    }

    auto *declRef = llvm::dyn_cast_or_null<clang::DeclRefExpr>(
        utils::ignoreExprWrappers(lhsExpr));

    if (declRef) {
      if (auto *varDecl = llvm::dyn_cast<clang::VarDecl>(declRef->getDecl())) {
        mlir::Type targetType = convertType(varDecl->getType());
        if (targetType) {
          resultValue =
              utils::promoteValue(builder, loc, resultValue, targetType);
          if (!resultValue) {
            return nullptr;
          }
        }

        localValueTable[varDecl] = resultValue;
        return resultValue;
      }
    }

    llvm::WithColor::error()
        << "chwc: unsupported overloaded compound assignment lhs\n";
    return resultValue;
  }

  if (op == OO::OO_PlusPlus || op == OO::OO_MinusMinus) {
    clang::Expr *valueExpr = getOperatorLHS(callExpr);
    if (!valueExpr) {
      llvm::WithColor::error()
          << "chwc: malformed overloaded increment/decrement operator\n";
      return nullptr;
    }

    bool isIncrement = op == OO::OO_PlusPlus;
    bool isPrefix = callExpr->getNumArgs() == 1;

    if (callExpr->getNumArgs() != 1 && callExpr->getNumArgs() != 2) {
      llvm::WithColor::error()
          << "chwc: unsupported overloaded increment/decrement arg count\n";
      return nullptr;
    }

    return generateIncDecUnaryOperator(valueExpr, isIncrement, isPrefix);
  }

  if (callExpr->getNumArgs() == 1) {
    mlir::Value value = generateExpr(callExpr->getArg(0));
    if (!value) {
      return nullptr;
    }

    switch (op) {
    case OO::OO_Tilde:
      return utils::bitNot(builder, loc, value);

    case OO::OO_Exclaim:
      return utils::icmpEq(builder, loc, utils::toBool(builder, loc, value),
                           utils::zeroValue(builder, loc, builder.getI1Type()));

    case OO::OO_Plus:
      return value;

    case OO::OO_Minus:
      return utils::neg(builder, loc, value);

    default:
      break;
    }
  }

  if (callExpr->getNumArgs() != 2) {
    llvm::WithColor::error()
        << "chwc: unsupported overloaded operator arg count\n";
    return nullptr;
  }

  mlir::Value lhs = generateExpr(callExpr->getArg(0));
  mlir::Value rhs = generateExpr(callExpr->getArg(1));

  if (!lhs || !rhs) {
    llvm::WithColor::error()
        << "chwc: failed to generate overloaded operator operands\n";
    return nullptr;
  }

  mlir::Type computeType = lhs.getType();
  if (computeType != rhs.getType()) {
    rhs = utils::promoteValue(builder, loc, rhs, computeType);
  }

  clang::Expr *objectExpr = callExpr->getArg(0);
  clang::QualType objectType =
      objectExpr ? objectExpr->getType() : clang::QualType{};
  auto typeInfo = utils::getSignalTypeInfo(objectType);

  switch (op) {
  case OO::OO_Plus:
    return utils::add(builder, loc, lhs, rhs);

  case OO::OO_Minus:
    return utils::sub(builder, loc, lhs, rhs);

  case OO::OO_Star:
    return utils::mul(builder, loc, lhs, rhs);

  case OO::OO_Slash:
    return typeInfo.isSigned ? utils::divS(builder, loc, lhs, rhs)
                             : utils::divU(builder, loc, lhs, rhs);

  case OO::OO_Percent:
    return typeInfo.isSigned ? utils::modS(builder, loc, lhs, rhs)
                             : utils::modU(builder, loc, lhs, rhs);

  case OO::OO_Amp:
    return utils::bitAnd(builder, loc, lhs, rhs);

  case OO::OO_Pipe:
    return utils::bitOr(builder, loc, lhs, rhs);

  case OO::OO_Caret:
    return utils::bitXor(builder, loc, lhs, rhs);

  case OO::OO_LessLess:
    return utils::shl(builder, loc, lhs, rhs);

  case OO::OO_GreaterGreater:
    return typeInfo.isSigned ? utils::shrS(builder, loc, lhs, rhs)
                             : utils::shrU(builder, loc, lhs, rhs);

  case OO::OO_EqualEqual:
    return utils::icmpEq(builder, loc, lhs, rhs);

  case OO::OO_ExclaimEqual:
    return utils::icmpNe(builder, loc, lhs, rhs);

  case OO::OO_Less:
    return typeInfo.isSigned ? utils::icmpSlt(builder, loc, lhs, rhs)
                             : utils::icmpUlt(builder, loc, lhs, rhs);

  case OO::OO_LessEqual:
    return typeInfo.isSigned ? utils::icmpSle(builder, loc, lhs, rhs)
                             : utils::icmpUle(builder, loc, lhs, rhs);

  case OO::OO_Greater:
    return typeInfo.isSigned ? utils::icmpSgt(builder, loc, lhs, rhs)
                             : utils::icmpUgt(builder, loc, lhs, rhs);

  case OO::OO_GreaterEqual:
    return typeInfo.isSigned ? utils::icmpSge(builder, loc, lhs, rhs)
                             : utils::icmpUge(builder, loc, lhs, rhs);

  case OO::OO_AmpAmp:
    return generateLAndBinaryOperator(lhs, rhs);

  case OO::OO_PipePipe:
    return generateLOrBinaryOperator(lhs, rhs);

  default:
    llvm::WithColor::error() << "chwc: unsupported overloaded operator\n";
    return nullptr;
  }
}

} // namespace chwc
