#include "../../Converter.h"
#include "../Utils/Array.h"
#include "../Utils/Cast.h"
#include "../Utils/Comb.h"
#include "../Utils/Expr.h"
#include "../Utils/Type.h"
#include "llvm/Support/WithColor.h"

namespace chwc {

auto CHWConverter::generateAssignmentBinaryOperator(
    clang::BinaryOperator *assignOp) -> mlir::Value {
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
        llvm::WithColor::error()
            << "chwc: reading output field is not supported\n";
        return nullptr;
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
      if (targetType) {
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
              << "chwc: reset assignment only supports Reg fields\n";
          return value;
        }

        fieldInfo.resetValue = value;
        return value;
      }

      switch (fieldInfo.kind) {
      case HWFieldKind::Input:
        llvm::WithColor::error() << "chwc: cannot assign to hardware input\n";
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
            << "chwc: array assignment target is not array field\n";
        return value;
      }

      value = utils::promoteValue(builder, loc, value, fieldInfo.elementType);
      if (!value) {
        return nullptr;
      }

      if (emitMode == HWEmitMode::Reset) {
        if (fieldInfo.kind != HWFieldKind::Reg) {
          llvm::WithColor::error()
              << "chwc: reset assignment only supports Reg array fields\n";
          return value;
        }

        if (!fieldInfo.resetValue) {
          fieldInfo.resetValue = utils::zeroArray(builder, loc, fieldInfo.type);
        }

        fieldInfo.resetValue = utils::arrayInject(
            builder, loc, fieldInfo.resetValue, target.index, value);
        return value;
      }

      mlir::Value arrayValue = nullptr;

      switch (fieldInfo.kind) {
      case HWFieldKind::Input:
        llvm::WithColor::error()
            << "chwc: cannot assign to hardware input array\n";
        return value;

      case HWFieldKind::Output:
        arrayValue = moduleContext.outputValues.lookup(target.fieldDecl);
        if (!arrayValue) {
          arrayValue = utils::zeroArray(builder, loc, fieldInfo.type);
        }

        moduleContext.outputValues[target.fieldDecl] =
            utils::arrayInject(builder, loc, arrayValue, target.index, value);
        return value;

      case HWFieldKind::Wire:
        arrayValue = moduleContext.currentValues.lookup(target.fieldDecl);
        if (!arrayValue) {
          arrayValue = utils::zeroArray(builder, loc, fieldInfo.type);
        }

        moduleContext.currentValues[target.fieldDecl] =
            utils::arrayInject(builder, loc, arrayValue, target.index, value);
        return value;

      case HWFieldKind::Reg:
        arrayValue = moduleContext.nextValues.lookup(target.fieldDecl);
        if (!arrayValue) {
          arrayValue = moduleContext.currentValues.lookup(target.fieldDecl);
        }

        moduleContext.nextValues[target.fieldDecl] =
            utils::arrayInject(builder, loc, arrayValue, target.index, value);
        return value;
      }

      return value;
    }

    case AssignTarget::Kind::Invalid:
      llvm::WithColor::error() << "chwc: invalid assignment target\n";
      return value;
    }

    return value;
  };

  auto emitCompound = [&](clang::BinaryOperatorKind op, mlir::Value lhs,
                          mlir::Value rhs, bool isSigned) -> mlir::Value {
    mlir::OpBuilder &builder = contextManager.Builder();
    mlir::Location loc = builder.getUnknownLoc();

    switch (op) {
    case clang::BO_AddAssign:
      return utils::add(builder, loc, lhs, rhs);
    case clang::BO_SubAssign:
      return utils::sub(builder, loc, lhs, rhs);
    case clang::BO_MulAssign:
      return utils::mul(builder, loc, lhs, rhs);
    case clang::BO_DivAssign:
      return isSigned ? utils::divS(builder, loc, lhs, rhs)
                      : utils::divU(builder, loc, lhs, rhs);
    case clang::BO_RemAssign:
      return isSigned ? utils::modS(builder, loc, lhs, rhs)
                      : utils::modU(builder, loc, lhs, rhs);
    case clang::BO_AndAssign:
      return utils::bitAnd(builder, loc, lhs, rhs);
    case clang::BO_OrAssign:
      return utils::bitOr(builder, loc, lhs, rhs);
    case clang::BO_XorAssign:
      return utils::bitXor(builder, loc, lhs, rhs);
    case clang::BO_ShlAssign:
      return utils::shl(builder, loc, lhs, rhs);
    case clang::BO_ShrAssign:
      return isSigned ? utils::shrS(builder, loc, lhs, rhs)
                      : utils::shrU(builder, loc, lhs, rhs);
    default:
      llvm::WithColor::error() << "chwc: unsupported compound assignment\n";
      return nullptr;
    }
  };

  std::optional<AssignTarget> target = resolveTarget(assignOp->getLHS());
  if (!target) {
    llvm::WithColor::error() << "chwc: unsupported assignment lhs\n";
    return nullptr;
  }

  mlir::Value rhs = generateExpr(assignOp->getRHS());
  if (!rhs) {
    llvm::WithColor::error() << "chwc: failed to generate RHS\n";
    return nullptr;
  }

  if (assignOp->getOpcode() == clang::BO_Assign) {
    return writeTarget(*target, rhs);
  }

  mlir::Value oldValue = readTarget(*target);
  if (!oldValue) {
    llvm::WithColor::error()
        << "chwc: failed to read compound assignment lhs\n";
    return nullptr;
  }

  mlir::OpBuilder &builder = contextManager.Builder();
  mlir::Location loc = builder.getUnknownLoc();

  rhs = utils::promoteValue(builder, loc, rhs, oldValue.getType());
  if (!rhs) {
    return nullptr;
  }

  bool isSigned = utils::isSignedType(assignOp->getLHS()->getType()) ||
                  utils::isSignedType(assignOp->getRHS()->getType());

  mlir::Value result =
      emitCompound(assignOp->getOpcode(), oldValue, rhs, isSigned);

  return writeTarget(*target, result);
}

} // namespace chwc
