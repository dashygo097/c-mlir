#include "../../Converter.h"
#include "../Utils/Cast.h"
#include "../Utils/Comb.h"
#include "../Utils/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "llvm/Support/WithColor.h"

namespace chwc {

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

    const clang::FieldDecl *fieldDecl = getAssignedField(lhsExpr);
    if (fieldDecl) {
      auto fieldIt = fieldTable.find(fieldDecl);
      if (fieldIt == fieldTable.end()) {
        llvm::WithColor::error()
            << "chwc: assignment lhs is not hardware field\n";
        return rhsValue;
      }

      HWFieldInfo &fieldInfo = fieldIt->second;

      rhsValue = utils::promoteValue(builder, loc, rhsValue, fieldInfo.type);
      if (!rhsValue) {
        return nullptr;
      }

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

    auto *declRef =
        mlir::dyn_cast_or_null<clang::DeclRefExpr>(utils::ignoreCasts(lhsExpr));
    if (declRef) {
      if (auto *varDecl = mlir::dyn_cast<clang::VarDecl>(declRef->getDecl())) {
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

  switch (op) {
  case OO::OO_Plus:
    return utils::add(builder, loc, lhs, rhs);

  case OO::OO_Minus:
    return utils::sub(builder, loc, lhs, rhs);

  case OO::OO_Star:
    return utils::mul(builder, loc, lhs, rhs);

  case OO::OO_Slash:
    return utils::div(builder, loc, lhs, rhs);

  case OO::OO_Percent:
    return utils::mod(builder, loc, lhs, rhs);

  case OO::OO_Amp:
    return utils::bitAnd(builder, loc, lhs, rhs);

  case OO::OO_Pipe:
    return utils::bitOr(builder, loc, lhs, rhs);

  case OO::OO_Caret:
    return utils::bitXor(builder, loc, lhs, rhs);

  case OO::OO_LessLess:
    return utils::shl(builder, loc, lhs, rhs);

  case OO::OO_GreaterGreater:
    return utils::shrU(builder, loc, lhs, rhs);

  case OO::OO_EqualEqual:
    return utils::icmpEq(builder, loc, lhs, rhs);

  case OO::OO_ExclaimEqual:
    return utils::icmpNe(builder, loc, lhs, rhs);

  case OO::OO_Less:
    return utils::icmpUlt(builder, loc, lhs, rhs);

  case OO::OO_LessEqual:
    return utils::icmpUle(builder, loc, lhs, rhs);

  case OO::OO_Greater:
    return utils::icmpUgt(builder, loc, lhs, rhs);

  case OO::OO_GreaterEqual:
    return utils::icmpUge(builder, loc, lhs, rhs);

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
