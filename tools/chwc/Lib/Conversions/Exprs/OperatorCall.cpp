#include "../../Converter.h"
#include "../Utils/Array.h"
#include "../Utils/Cast.h"
#include "../Utils/Comb.h"
#include "../Utils/Expr.h"
#include "../Utils/Type.h"
#include "llvm/Support/WithColor.h"

namespace chwc {

auto CHWConverter::generateCXXOperatorCallExpr(
    clang::CXXOperatorCallExpr *callExpr) -> mlir::Value {
  if (!callExpr) {
    return nullptr;
  }

  clang::OverloadedOperatorKind op = callExpr->getOperator();

  if (op == clang::OO_Equal) {
    if (callExpr->getNumArgs() != 2) {
      llvm::WithColor::error()
          << "chwc: overloaded operator= expects 2 operands\n";
      return nullptr;
    }

    clang::Expr *lhsExpr = callExpr->getArg(0);
    clang::Expr *rhsExpr = callExpr->getArg(1);

    mlir::Value rhsValue = generateExpr(rhsExpr);
    if (!rhsValue) {
      llvm::WithColor::error()
          << "chwc: failed to generate RHS for overloaded assignment\n";
      return nullptr;
    }

    mlir::OpBuilder &builder = contextManager.Builder();
    mlir::Location loc = builder.getUnknownLoc();

    clang::Expr *lhs = utils::ignoreCasts(lhsExpr);

    if (auto *arraySub =
            mlir::dyn_cast_or_null<clang::ArraySubscriptExpr>(lhs)) {
      clang::Expr *base = utils::ignoreCasts(arraySub->getBase());

      const clang::FieldDecl *fieldDecl = nullptr;

      if (auto *memberExpr = mlir::dyn_cast_or_null<clang::MemberExpr>(base)) {
        fieldDecl =
            mlir::dyn_cast<clang::FieldDecl>(memberExpr->getMemberDecl());
      } else if (auto *declRef =
                     mlir::dyn_cast_or_null<clang::DeclRefExpr>(base)) {
        fieldDecl = mlir::dyn_cast<clang::FieldDecl>(declRef->getDecl());
      }

      if (!fieldDecl) {
        llvm::WithColor::error()
            << "chwc: unsupported overloaded assignment array lhs\n";
        return rhsValue;
      }

      auto fieldIt = moduleContext.fields.find(fieldDecl);
      if (fieldIt == moduleContext.fields.end() || !fieldIt->second.isArray) {
        llvm::WithColor::error()
            << "chwc: overloaded assignment lhs is not hardware array field\n";
        return rhsValue;
      }

      HWFieldInfo &fieldInfo = fieldIt->second;

      mlir::Value index = generateExpr(arraySub->getIdx());
      if (!index) {
        return rhsValue;
      }

      rhsValue =
          utils::promoteValue(builder, loc, rhsValue, fieldInfo.elementType);
      if (!rhsValue) {
        return nullptr;
      }

      if (emitMode == HWEmitMode::Reset) {
        if (fieldInfo.kind != HWFieldKind::Reg) {
          llvm::WithColor::error()
              << "chwc: reset assignment only supports Reg array\n";
          return rhsValue;
        }

        if (!fieldInfo.resetValue) {
          fieldInfo.resetValue = utils::zeroArray(builder, loc, fieldInfo.type);
        }

        fieldInfo.resetValue = utils::arrayInject(
            builder, loc, fieldInfo.resetValue, index, rhsValue);
        return rhsValue;
      }

      mlir::Value oldArray = nullptr;

      switch (fieldInfo.kind) {
      case HWFieldKind::Input:
        llvm::WithColor::error()
            << "chwc: cannot assign to hardware input array\n";
        return rhsValue;

      case HWFieldKind::Output:
        oldArray = moduleContext.outputValues.lookup(fieldDecl);
        if (!oldArray) {
          oldArray = utils::zeroArray(builder, loc, fieldInfo.type);
        }

        moduleContext.outputValues[fieldDecl] =
            utils::arrayInject(builder, loc, oldArray, index, rhsValue);
        return rhsValue;

      case HWFieldKind::Wire:
        oldArray = moduleContext.currentValues.lookup(fieldDecl);
        if (!oldArray) {
          oldArray = utils::zeroArray(builder, loc, fieldInfo.type);
        }

        moduleContext.currentValues[fieldDecl] =
            utils::arrayInject(builder, loc, oldArray, index, rhsValue);
        return rhsValue;

      case HWFieldKind::Reg:
        oldArray = moduleContext.nextValues.lookup(fieldDecl);
        if (!oldArray) {
          oldArray = moduleContext.currentValues.lookup(fieldDecl);
        }

        moduleContext.nextValues[fieldDecl] =
            utils::arrayInject(builder, loc, oldArray, index, rhsValue);
        return rhsValue;
      }
    }

    if (auto *memberExpr = mlir::dyn_cast_or_null<clang::MemberExpr>(lhs)) {
      if (auto *fieldDecl =
              mlir::dyn_cast<clang::FieldDecl>(memberExpr->getMemberDecl())) {
        auto fieldIt = moduleContext.fields.find(fieldDecl);
        if (fieldIt == moduleContext.fields.end()) {
          llvm::WithColor::error()
              << "chwc: assignment lhs is not hardware field\n";
          return rhsValue;
        }

        HWFieldInfo &fieldInfo = fieldIt->second;

        rhsValue = utils::promoteValue(builder, loc, rhsValue, fieldInfo.type);
        if (!rhsValue) {
          return nullptr;
        }

        if (emitMode == HWEmitMode::Reset) {
          if (fieldInfo.kind != HWFieldKind::Reg) {
            llvm::WithColor::error()
                << "chwc: reset assignment only supports Reg field\n";
            return rhsValue;
          }

          fieldInfo.resetValue = rhsValue;
          return rhsValue;
        }

        switch (fieldInfo.kind) {
        case HWFieldKind::Input:
          llvm::WithColor::error() << "chwc: cannot assign to hardware input\n";
          break;

        case HWFieldKind::Output:
          moduleContext.outputValues[fieldDecl] = rhsValue;
          break;

        case HWFieldKind::Wire:
          moduleContext.currentValues[fieldDecl] = rhsValue;
          break;

        case HWFieldKind::Reg:
          moduleContext.nextValues[fieldDecl] = rhsValue;
          break;
        }

        return rhsValue;
      }
    }

    if (auto *declRef = mlir::dyn_cast_or_null<clang::DeclRefExpr>(lhs)) {
      if (auto *fieldDecl =
              mlir::dyn_cast<clang::FieldDecl>(declRef->getDecl())) {
        auto fieldIt = moduleContext.fields.find(fieldDecl);
        if (fieldIt == moduleContext.fields.end()) {
          llvm::WithColor::error()
              << "chwc: assignment lhs is not hardware field\n";
          return rhsValue;
        }

        HWFieldInfo &fieldInfo = fieldIt->second;

        rhsValue = utils::promoteValue(builder, loc, rhsValue, fieldInfo.type);
        if (!rhsValue) {
          return nullptr;
        }

        if (emitMode == HWEmitMode::Reset) {
          if (fieldInfo.kind != HWFieldKind::Reg) {
            llvm::WithColor::error()
                << "chwc: reset assignment only supports Reg field\n";
            return rhsValue;
          }

          fieldInfo.resetValue = rhsValue;
          return rhsValue;
        }

        switch (fieldInfo.kind) {
        case HWFieldKind::Input:
          llvm::WithColor::error() << "chwc: cannot assign to hardware input\n";
          break;

        case HWFieldKind::Output:
          moduleContext.outputValues[fieldDecl] = rhsValue;
          break;

        case HWFieldKind::Wire:
          moduleContext.currentValues[fieldDecl] = rhsValue;
          break;

        case HWFieldKind::Reg:
          moduleContext.nextValues[fieldDecl] = rhsValue;
          break;
        }

        return rhsValue;
      }

      if (auto *varDecl = mlir::dyn_cast<clang::VarDecl>(declRef->getDecl())) {
        if (functionStack.empty()) {
          functionStack.emplace_back();
        }

        mlir::Type targetType = convertType(varDecl->getType());
        if (targetType) {
          rhsValue = utils::promoteValue(builder, loc, rhsValue, targetType);
          if (!rhsValue) {
            return nullptr;
          }
        }

        functionStack.back().locals[varDecl] = rhsValue;
        return rhsValue;
      }
    }

    llvm::WithColor::error() << "chwc: unsupported overloaded assignment lhs\n";
    return rhsValue;
  }

  if (callExpr->getNumArgs() == 1) {
    mlir::Value operand = generateExpr(callExpr->getArg(0));
    if (!operand) {
      llvm::WithColor::error()
          << "chwc: failed to generate overloaded unary operand\n";
      return nullptr;
    }

    mlir::OpBuilder &builder = contextManager.Builder();
    mlir::Location loc = builder.getUnknownLoc();

    switch (op) {
    case clang::OO_Exclaim:
      return utils::icmpEq(builder, loc, utils::toBool(builder, loc, operand),
                           utils::boolConst(builder, loc, false));

    case clang::OO_Tilde:
      return utils::bitXor(
          builder, loc, operand,
          utils::intConst(builder, loc, operand.getType(), -1));

    default:
      llvm::WithColor::error()
          << "chwc: unsupported overloaded unary operator\n";
      return nullptr;
    }
  }

  if (callExpr->getNumArgs() != 2) {
    llvm::WithColor::error()
        << "chwc: unsupported overloaded operator operand count\n";
    return nullptr;
  }

  mlir::Value lhsValue = generateExpr(callExpr->getArg(0));
  mlir::Value rhsValue = generateExpr(callExpr->getArg(1));

  if (!lhsValue || !rhsValue) {
    llvm::WithColor::error()
        << "chwc: failed to generate overloaded operator operands\n";
    return nullptr;
  }

  mlir::OpBuilder &builder = contextManager.Builder();
  mlir::Location loc = builder.getUnknownLoc();

  if (lhsValue.getType() != rhsValue.getType()) {
    rhsValue = utils::promoteValue(builder, loc, rhsValue, lhsValue.getType());
    if (!rhsValue) {
      return nullptr;
    }
  }

  bool isSigned = utils::isSignedType(callExpr->getArg(0)->getType()) ||
                  utils::isSignedType(callExpr->getArg(1)->getType());

  bool isSignedShift = utils::isSignedType(callExpr->getArg(0)->getType());

  switch (op) {
  case clang::OO_Plus:
    return utils::add(builder, loc, lhsValue, rhsValue);

  case clang::OO_Minus:
    return utils::sub(builder, loc, lhsValue, rhsValue);

  case clang::OO_Star:
    return utils::mul(builder, loc, lhsValue, rhsValue);

  case clang::OO_Slash:
    return isSigned ? utils::divS(builder, loc, lhsValue, rhsValue)
                    : utils::divU(builder, loc, lhsValue, rhsValue);

  case clang::OO_Percent:
    return isSigned ? utils::modS(builder, loc, lhsValue, rhsValue)
                    : utils::modU(builder, loc, lhsValue, rhsValue);

  case clang::OO_Amp:
    return utils::bitAnd(builder, loc, lhsValue, rhsValue);

  case clang::OO_Pipe:
    return utils::bitOr(builder, loc, lhsValue, rhsValue);

  case clang::OO_Caret:
    return utils::bitXor(builder, loc, lhsValue, rhsValue);

  case clang::OO_LessLess:
    return utils::shl(builder, loc, lhsValue, rhsValue);

  case clang::OO_GreaterGreater:
    return isSignedShift ? utils::shrS(builder, loc, lhsValue, rhsValue)
                         : utils::shrU(builder, loc, lhsValue, rhsValue);

  case clang::OO_EqualEqual:
    return utils::icmpEq(builder, loc, lhsValue, rhsValue);

  case clang::OO_ExclaimEqual:
    return utils::icmpNe(builder, loc, lhsValue, rhsValue);

  case clang::OO_Less:
    return isSigned ? utils::icmpSlt(builder, loc, lhsValue, rhsValue)
                    : utils::icmpUlt(builder, loc, lhsValue, rhsValue);

  case clang::OO_LessEqual:
    return isSigned ? utils::icmpSle(builder, loc, lhsValue, rhsValue)
                    : utils::icmpUle(builder, loc, lhsValue, rhsValue);

  case clang::OO_Greater:
    return isSigned ? utils::icmpSgt(builder, loc, lhsValue, rhsValue)
                    : utils::icmpUgt(builder, loc, lhsValue, rhsValue);

  case clang::OO_GreaterEqual:
    return isSigned ? utils::icmpSge(builder, loc, lhsValue, rhsValue)
                    : utils::icmpUge(builder, loc, lhsValue, rhsValue);

  case clang::OO_AmpAmp:
    return generateLAndBinaryOperator(lhsValue, rhsValue);

  case clang::OO_PipePipe:
    return generateLOrBinaryOperator(lhsValue, rhsValue);

  default:
    llvm::WithColor::error() << "chwc: unsupported overloaded operator\n";
    return nullptr;
  }
}

} // namespace chwc
