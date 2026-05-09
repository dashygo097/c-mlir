#include "../../Converter.h"
#include "../Utils/Array.h"
#include "../Utils/Cast.h"
#include "../Utils/Comb.h"
#include "../Utils/Constant.h"
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

  struct AssignTarget {
    enum class Kind {
      Invalid,
      Local,
      Field,
      ArrayElement,
    };

    Kind kind{Kind::Invalid};
    const clang::VarDecl *localDecl{nullptr};
    const clang::FieldDecl *fieldDecl{nullptr};
    mlir::Value index{};

    explicit operator bool() const { return kind != Kind::Invalid; }
  };

  auto resolveTarget = [&](clang::Expr *expr) -> std::optional<AssignTarget> {
    expr = utils::ignoreCasts(expr);

    if (auto *arraySub =
            mlir::dyn_cast_or_null<clang::ArraySubscriptExpr>(expr)) {
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
        return std::nullopt;
      }

      auto fieldIt = moduleContext.fields.find(fieldDecl);
      if (fieldIt == moduleContext.fields.end() || !fieldIt->second.isArray) {
        return std::nullopt;
      }

      mlir::Value index = generateExpr(arraySub->getIdx());
      if (!index) {
        return std::nullopt;
      }

      AssignTarget target;
      target.kind = AssignTarget::Kind::ArrayElement;
      target.fieldDecl = fieldDecl;
      target.index = index;
      return target;
    }

    if (auto *memberExpr = mlir::dyn_cast_or_null<clang::MemberExpr>(expr)) {
      if (auto *fieldDecl =
              mlir::dyn_cast<clang::FieldDecl>(memberExpr->getMemberDecl())) {
        AssignTarget target;
        target.kind = AssignTarget::Kind::Field;
        target.fieldDecl = fieldDecl;
        return target;
      }
    }

    if (auto *declRef = mlir::dyn_cast_or_null<clang::DeclRefExpr>(expr)) {
      if (auto *fieldDecl =
              mlir::dyn_cast<clang::FieldDecl>(declRef->getDecl())) {
        AssignTarget target;
        target.kind = AssignTarget::Kind::Field;
        target.fieldDecl = fieldDecl;
        return target;
      }

      if (auto *varDecl = mlir::dyn_cast<clang::VarDecl>(declRef->getDecl())) {
        AssignTarget target;
        target.kind = AssignTarget::Kind::Local;
        target.localDecl = varDecl;
        return target;
      }
    }

    return std::nullopt;
  };

  auto getTargetType = [&](const AssignTarget &target) -> mlir::Type {
    switch (target.kind) {
    case AssignTarget::Kind::Local:
      return convertType(target.localDecl->getType());

    case AssignTarget::Kind::Field: {
      auto fieldIt = moduleContext.fields.find(target.fieldDecl);
      if (fieldIt == moduleContext.fields.end()) {
        return nullptr;
      }

      return fieldIt->second.type;
    }

    case AssignTarget::Kind::ArrayElement: {
      auto fieldIt = moduleContext.fields.find(target.fieldDecl);
      if (fieldIt == moduleContext.fields.end()) {
        return nullptr;
      }

      return fieldIt->second.elementType;
    }

    case AssignTarget::Kind::Invalid:
      return nullptr;
    }

    return nullptr;
  };

  auto generateRHSForTarget = [&](clang::Expr *rhsExpr,
                                  mlir::Type targetType) -> mlir::Value {
    if (!rhsExpr || !targetType) {
      return nullptr;
    }

    rhsExpr = rhsExpr->IgnoreParenImpCasts();

    mlir::OpBuilder &builder = contextManager.Builder();
    mlir::Location loc = builder.getUnknownLoc();

    if (auto *intLit = mlir::dyn_cast<clang::IntegerLiteral>(rhsExpr)) {
      return utils::intConst(builder, loc, targetType,
                             intLit->getValue().getSExtValue());
    }

    mlir::Value value = generateExpr(rhsExpr);
    if (!value) {
      return nullptr;
    }

    if (value.getType() == targetType) {
      return value;
    }

    return utils::promoteValue(builder, loc, value, targetType);
  };

  auto readTarget = [&](const AssignTarget &target) -> mlir::Value {
    switch (target.kind) {
    case AssignTarget::Kind::Local:
      if (functionStack.empty()) {
        return nullptr;
      }

      return functionStack.back().locals.lookup(target.localDecl);

    case AssignTarget::Kind::Field: {
      auto fieldIt = moduleContext.fields.find(target.fieldDecl);
      if (fieldIt == moduleContext.fields.end()) {
        return nullptr;
      }

      if (fieldIt->second.kind == HWFieldKind::Output) {
        return moduleContext.outputValues.lookup(target.fieldDecl);
      }

      return moduleContext.currentValues.lookup(target.fieldDecl);
    }

    case AssignTarget::Kind::ArrayElement: {
      auto fieldIt = moduleContext.fields.find(target.fieldDecl);
      if (fieldIt == moduleContext.fields.end()) {
        return nullptr;
      }

      mlir::Value arrayValue =
          moduleContext.currentValues.lookup(target.fieldDecl);
      if (!arrayValue) {
        return nullptr;
      }

      mlir::OpBuilder &builder = contextManager.Builder();
      mlir::Location loc = builder.getUnknownLoc();

      return utils::arrayGet(builder, loc, arrayValue, target.index);
    }

    case AssignTarget::Kind::Invalid:
      return nullptr;
    }

    return nullptr;
  };

  auto writeTarget = [&](const AssignTarget &target,
                         mlir::Value value) -> mlir::Value {
    if (!value) {
      return nullptr;
    }

    mlir::OpBuilder &builder = contextManager.Builder();
    mlir::Location loc = builder.getUnknownLoc();

    switch (target.kind) {
    case AssignTarget::Kind::Local: {
      if (functionStack.empty()) {
        functionStack.emplace_back();
      }

      mlir::Type targetType = convertType(target.localDecl->getType());
      if (targetType && value.getType() != targetType) {
        value = utils::promoteValue(builder, loc, value, targetType);
        if (!value) {
          return nullptr;
        }
      }

      functionStack.back().locals[target.localDecl] = value;
      return value;
    }

    case AssignTarget::Kind::Field: {
      auto fieldIt = moduleContext.fields.find(target.fieldDecl);
      if (fieldIt == moduleContext.fields.end()) {
        llvm::WithColor::error()
            << "chwc: assignment lhs is not hardware field\n";
        return value;
      }

      HWFieldInfo &fieldInfo = fieldIt->second;

      if (value.getType() != fieldInfo.type) {
        value = utils::promoteValue(builder, loc, value, fieldInfo.type);
        if (!value) {
          return nullptr;
        }
      }

      if (emitMode == HWEmitMode::Reset) {
        if (fieldInfo.kind != HWFieldKind::Reg) {
          llvm::WithColor::error()
              << "chwc: reset assignment only supports Reg field\n";
          return value;
        }

        fieldInfo.resetValue = value;
        return value;
      }

      switch (fieldInfo.kind) {
      case HWFieldKind::Input:
        llvm::WithColor::error() << "chwc: cannot assign to module input\n";
        break;

      case HWFieldKind::Output:
        moduleContext.outputValues[target.fieldDecl] = value;
        break;

      case HWFieldKind::Wire:
        moduleContext.currentValues[target.fieldDecl] = value;
        break;

      case HWFieldKind::Reg:
        moduleContext.nextValues[target.fieldDecl] = value;
        break;
      }

      return value;
    }

    case AssignTarget::Kind::ArrayElement: {
      auto fieldIt = moduleContext.fields.find(target.fieldDecl);
      if (fieldIt == moduleContext.fields.end()) {
        llvm::WithColor::error() << "chwc: unknown hardware array field\n";
        return value;
      }

      HWFieldInfo &fieldInfo = fieldIt->second;
      if (!fieldInfo.isArray) {
        llvm::WithColor::error()
            << "chwc: overloaded assignment lhs is not hardware array field\n";
        return value;
      }

      if (value.getType() != fieldInfo.elementType) {
        value = utils::promoteValue(builder, loc, value, fieldInfo.elementType);
        if (!value) {
          return nullptr;
        }
      }

      if (emitMode == HWEmitMode::Reset) {
        if (fieldInfo.kind != HWFieldKind::Reg) {
          llvm::WithColor::error()
              << "chwc: reset assignment only supports Reg array\n";
          return value;
        }

        if (!fieldInfo.resetValue) {
          fieldInfo.resetValue = utils::zeroArray(builder, loc, fieldInfo.type);
        }

        fieldInfo.resetValue = utils::arrayInject(
            builder, loc, fieldInfo.resetValue, target.index, value);
        return value;
      }

      mlir::Value oldArray = nullptr;

      switch (fieldInfo.kind) {
      case HWFieldKind::Input:
        llvm::WithColor::error()
            << "chwc: cannot assign to module input array\n";
        return value;

      case HWFieldKind::Output:
        oldArray = moduleContext.outputValues.lookup(target.fieldDecl);
        if (!oldArray) {
          oldArray = utils::zeroArray(builder, loc, fieldInfo.type);
        }

        moduleContext.outputValues[target.fieldDecl] =
            utils::arrayInject(builder, loc, oldArray, target.index, value);
        return value;

      case HWFieldKind::Wire:
        oldArray = moduleContext.currentValues.lookup(target.fieldDecl);
        if (!oldArray) {
          oldArray = utils::zeroArray(builder, loc, fieldInfo.type);
        }

        moduleContext.currentValues[target.fieldDecl] =
            utils::arrayInject(builder, loc, oldArray, target.index, value);
        return value;

      case HWFieldKind::Reg:
        oldArray = moduleContext.nextValues.lookup(target.fieldDecl);
        if (!oldArray) {
          oldArray = moduleContext.currentValues.lookup(target.fieldDecl);
        }

        moduleContext.nextValues[target.fieldDecl] =
            utils::arrayInject(builder, loc, oldArray, target.index, value);
        return value;
      }

      return value;
    }

    case AssignTarget::Kind::Invalid:
      llvm::WithColor::error()
          << "chwc: unsupported overloaded assignment lhs\n";
      return value;
    }

    return value;
  };

  auto emitCompound = [&](clang::OverloadedOperatorKind compoundOp,
                          mlir::Value lhs, mlir::Value rhs,
                          bool isSigned) -> mlir::Value {
    mlir::OpBuilder &builder = contextManager.Builder();
    mlir::Location loc = builder.getUnknownLoc();

    switch (compoundOp) {
    case clang::OO_PlusEqual:
      return utils::add(builder, loc, lhs, rhs);
    case clang::OO_MinusEqual:
      return utils::sub(builder, loc, lhs, rhs);
    case clang::OO_StarEqual:
      return utils::mul(builder, loc, lhs, rhs);
    case clang::OO_SlashEqual:
      return isSigned ? utils::divS(builder, loc, lhs, rhs)
                      : utils::divU(builder, loc, lhs, rhs);
    case clang::OO_PercentEqual:
      return isSigned ? utils::modS(builder, loc, lhs, rhs)
                      : utils::modU(builder, loc, lhs, rhs);
    case clang::OO_AmpEqual:
      return utils::bitAnd(builder, loc, lhs, rhs);
    case clang::OO_PipeEqual:
      return utils::bitOr(builder, loc, lhs, rhs);
    case clang::OO_CaretEqual:
      return utils::bitXor(builder, loc, lhs, rhs);
    case clang::OO_LessLessEqual:
      return utils::shl(builder, loc, lhs, rhs);
    case clang::OO_GreaterGreaterEqual:
      return isSigned ? utils::shrS(builder, loc, lhs, rhs)
                      : utils::shrU(builder, loc, lhs, rhs);
    default:
      llvm::WithColor::error()
          << "chwc: unsupported overloaded compound assignment\n";
      return nullptr;
    }
  };

  auto isCompoundAssignOp = [](clang::OverloadedOperatorKind op) -> bool {
    switch (op) {
    case clang::OO_PlusEqual:
    case clang::OO_MinusEqual:
    case clang::OO_StarEqual:
    case clang::OO_SlashEqual:
    case clang::OO_PercentEqual:
    case clang::OO_AmpEqual:
    case clang::OO_PipeEqual:
    case clang::OO_CaretEqual:
    case clang::OO_LessLessEqual:
    case clang::OO_GreaterGreaterEqual:
      return true;
    default:
      return false;
    }
  };

  if (op == clang::OO_Equal || isCompoundAssignOp(op)) {
    if (callExpr->getNumArgs() != 2) {
      llvm::WithColor::error()
          << "chwc: overloaded assignment expects 2 operands\n";
      return nullptr;
    }

    std::optional<AssignTarget> target = resolveTarget(callExpr->getArg(0));
    if (!target) {
      llvm::WithColor::error()
          << "chwc: unsupported overloaded assignment lhs\n";
      return nullptr;
    }

    if (op == clang::OO_Equal) {
      mlir::Type targetType = getTargetType(*target);
      if (!targetType) {
        return nullptr;
      }

      mlir::Value rhs = generateRHSForTarget(callExpr->getArg(1), targetType);
      if (!rhs) {
        llvm::WithColor::error()
            << "chwc: failed to generate RHS for overloaded assignment\n";
        return nullptr;
      }

      return writeTarget(*target, rhs);
    }

    mlir::Value oldValue = readTarget(*target);
    if (!oldValue) {
      llvm::WithColor::error()
          << "chwc: failed to read overloaded compound assignment lhs\n";
      return nullptr;
    }

    mlir::Value rhs =
        generateRHSForTarget(callExpr->getArg(1), oldValue.getType());
    if (!rhs) {
      llvm::WithColor::error()
          << "chwc: failed to generate overloaded compound assignment rhs\n";
      return nullptr;
    }

    bool isSigned = utils::isSignedType(callExpr->getArg(0)->getType()) ||
                    utils::isSignedType(callExpr->getArg(1)->getType());

    mlir::Value result = emitCompound(op, oldValue, rhs, isSigned);
    if (!result) {
      return nullptr;
    }

    return writeTarget(*target, result);
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
  if (!lhsValue) {
    llvm::WithColor::error()
        << "chwc: failed to generate overloaded operator lhs\n";
    return nullptr;
  }

  mlir::Value rhsValue =
      generateRHSForTarget(callExpr->getArg(1), lhsValue.getType());
  if (!rhsValue) {
    llvm::WithColor::error()
        << "chwc: failed to generate overloaded operator rhs\n";
    return nullptr;
  }

  mlir::OpBuilder &builder = contextManager.Builder();
  mlir::Location loc = builder.getUnknownLoc();

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
