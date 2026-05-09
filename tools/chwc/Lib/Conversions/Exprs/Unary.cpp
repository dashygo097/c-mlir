#include "../../Converter.h"
#include "../Utils/Cast.h"
#include "../Utils/Comb.h"
#include "../Utils/Constant.h"
#include "../Utils/Expr.h"
#include "llvm/Support/WithColor.h"

namespace chwc {

auto CHWConverter::generateUnaryOperator(clang::UnaryOperator *unOp)
    -> mlir::Value {
  if (!unOp) {
    return nullptr;
  }

  mlir::OpBuilder &builder = contextManager.Builder();
  mlir::Location loc = builder.getUnknownLoc();

  clang::Expr *subExpr = unOp->getSubExpr();

  auto resolveScalarTarget = [&](clang::Expr *expr)
      -> std::optional<
          std::pair<const clang::VarDecl *, const clang::FieldDecl *>> {
    expr = utils::ignoreCasts(expr);

    if (auto *memberExpr = mlir::dyn_cast_or_null<clang::MemberExpr>(expr)) {
      if (auto *fieldDecl =
              mlir::dyn_cast<clang::FieldDecl>(memberExpr->getMemberDecl())) {
        return std::make_pair(nullptr, fieldDecl);
      }
    }

    if (auto *declRef = mlir::dyn_cast_or_null<clang::DeclRefExpr>(expr)) {
      if (auto *fieldDecl =
              mlir::dyn_cast<clang::FieldDecl>(declRef->getDecl())) {
        return std::make_pair(nullptr, fieldDecl);
      }

      if (auto *varDecl = mlir::dyn_cast<clang::VarDecl>(declRef->getDecl())) {
        return std::make_pair(varDecl, nullptr);
      }
    }

    return std::nullopt;
  };

  auto readScalarTarget =
      [&](const clang::VarDecl *varDecl,
          const clang::FieldDecl *fieldDecl) -> mlir::Value {
    if (varDecl) {
      if (functionStack.empty()) {
        return nullptr;
      }

      return functionStack.back().locals.lookup(varDecl);
    }

    if (fieldDecl) {
      auto fieldIt = moduleContext.fields.find(fieldDecl);
      if (fieldIt == moduleContext.fields.end()) {
        llvm::WithColor::error() << "chwc: unknown hardware field\n";
        return nullptr;
      }

      if (fieldIt->second.kind == HWFieldKind::Output) {
        llvm::WithColor::error()
            << "chwc: reading output field is not supported\n";
        return nullptr;
      }

      return moduleContext.currentValues.lookup(fieldDecl);
    }

    return nullptr;
  };

  auto writeScalarTarget = [&](const clang::VarDecl *varDecl,
                               const clang::FieldDecl *fieldDecl,
                               mlir::Value value) -> mlir::Value {
    if (!value) {
      return nullptr;
    }

    if (varDecl) {
      if (functionStack.empty()) {
        functionStack.emplace_back();
      }

      mlir::Type targetType = convertType(varDecl->getType());
      if (targetType) {
        value = utils::promoteValue(builder, loc, value, targetType);
        if (!value) {
          return nullptr;
        }
      }

      functionStack.back().locals[varDecl] = value;
      return value;
    }

    if (fieldDecl) {
      auto fieldIt = moduleContext.fields.find(fieldDecl);
      if (fieldIt == moduleContext.fields.end()) {
        llvm::WithColor::error() << "chwc: unknown hardware field\n";
        return value;
      }

      HWFieldInfo &fieldInfo = fieldIt->second;

      value = utils::promoteValue(builder, loc, value, fieldInfo.type);
      if (!value) {
        return nullptr;
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
        moduleContext.outputValues[fieldDecl] = value;
        break;

      case HWFieldKind::Wire:
        moduleContext.currentValues[fieldDecl] = value;
        break;

      case HWFieldKind::Reg:
        moduleContext.nextValues[fieldDecl] = value;
        break;
      }

      return value;
    }

    return value;
  };

  auto applyIncDec = [&](mlir::Value oldValue,
                         bool isIncrement) -> mlir::Value {
    if (!oldValue) {
      return nullptr;
    }

    mlir::Value one = utils::intConst(builder, loc, oldValue.getType(), 1);

    return isIncrement ? utils::add(builder, loc, oldValue, one)
                       : utils::sub(builder, loc, oldValue, one);
  };

  switch (unOp->getOpcode()) {
  case clang::UO_Plus:
    return generateExpr(subExpr);

  case clang::UO_Minus: {
    mlir::Value value = generateExpr(subExpr);
    if (!value) {
      return nullptr;
    }

    mlir::Value zero = utils::intConst(builder, loc, value.getType(), 0);
    return utils::sub(builder, loc, zero, value);
  }

  case clang::UO_LNot: {
    mlir::Value value = generateExpr(subExpr);
    if (!value) {
      return nullptr;
    }

    value = utils::toBool(builder, loc, value);
    if (!value) {
      return nullptr;
    }

    return utils::icmpEq(builder, loc, value,
                         utils::boolConst(builder, loc, false));
  }

  case clang::UO_Not: {
    mlir::Value value = generateExpr(subExpr);
    if (!value) {
      return nullptr;
    }

    return utils::bitXor(builder, loc, value,
                         utils::intConst(builder, loc, value.getType(), -1));
  }

  case clang::UO_PreInc:
  case clang::UO_PostInc:
  case clang::UO_PreDec:
  case clang::UO_PostDec: {
    bool isIncrement = unOp->getOpcode() == clang::UO_PreInc ||
                       unOp->getOpcode() == clang::UO_PostInc;
    bool isPrefix = unOp->getOpcode() == clang::UO_PreInc ||
                    unOp->getOpcode() == clang::UO_PreDec;

    std::optional<std::pair<const clang::VarDecl *, const clang::FieldDecl *>>
        target = resolveScalarTarget(subExpr);

    if (!target) {
      llvm::WithColor::error()
          << "chwc: increment/decrement only supports scalar local or field\n";
      return nullptr;
    }

    const clang::VarDecl *varDecl = target->first;
    const clang::FieldDecl *fieldDecl = target->second;

    mlir::Value oldValue = readScalarTarget(varDecl, fieldDecl);
    if (!oldValue) {
      llvm::WithColor::error()
          << "chwc: failed to read increment/decrement target\n";
      return nullptr;
    }

    mlir::Value newValue = applyIncDec(oldValue, isIncrement);
    if (!newValue) {
      return nullptr;
    }

    writeScalarTarget(varDecl, fieldDecl, newValue);

    return isPrefix ? newValue : oldValue;
  }

  default:
    llvm::WithColor::error()
        << "chwc: unsupported unary operator: "
        << clang::UnaryOperator::getOpcodeStr(unOp->getOpcode()) << "\n";
    return nullptr;
  }
}

} // namespace chwc
