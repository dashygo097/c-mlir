#include "../../../Converter.h"
#include "../../Utils/Comb.h"
#include "../../Utils/Constant.h"
#include "../../Utils/Expr.h"
#include "llvm/Support/WithColor.h"

namespace chwc {
auto applyIncDec(mlir::OpBuilder &builder, mlir::Location loc,
                 mlir::Value value, bool isIncrement) -> mlir::Value {
  if (!value) {
    return nullptr;
  }

  mlir::Value one = utils::oneValue(builder, loc, value.getType());
  if (!one) {
    return nullptr;
  }

  return isIncrement ? utils::add(builder, loc, value, one)
                     : utils::sub(builder, loc, value, one);
}

auto CHWConverter::generateIncDecUnaryOperator(clang::Expr *expr,
                                               bool isIncrement, bool isPrefix)
    -> mlir::Value {
  mlir::OpBuilder &builder = contextManager.Builder();
  mlir::Location loc = builder.getUnknownLoc();

  clang::Expr *bare = utils::ignoreCasts(expr);

  mlir::Value oldValue = generateExpr(expr);
  if (!oldValue) {
    llvm::WithColor::error()
        << "chwc: cannot read value for increment/decrement\n";
    return nullptr;
  }

  mlir::Value newValue = applyIncDec(builder, loc, oldValue, isIncrement);
  if (!newValue) {
    return nullptr;
  }

  const clang::FieldDecl *fieldDecl = getAssignedField(expr);
  if (fieldDecl) {
    auto fieldIt = fieldTable.find(fieldDecl);
    if (fieldIt == fieldTable.end()) {
      llvm::WithColor::error()
          << "chwc: increment/decrement lhs is not hardware field\n";
      return nullptr;
    }

    HWFieldInfo &fieldInfo = fieldIt->second;

    switch (fieldInfo.kind) {
    case HWFieldKind::Input:
      llvm::WithColor::error()
          << "chwc: cannot increment/decrement hardware input\n";
      return nullptr;

    case HWFieldKind::Output:
      outputValueTable[fieldDecl] = newValue;
      break;

    case HWFieldKind::Reg:
      nextFieldValueTable[fieldDecl] = newValue;
      break;

    case HWFieldKind::Wire:
      currentFieldValueTable[fieldDecl] = newValue;
      break;
    }

    return isPrefix ? newValue : oldValue;
  }

  if (auto *declRef = mlir::dyn_cast_or_null<clang::DeclRefExpr>(bare)) {
    if (auto *varDecl = mlir::dyn_cast<clang::VarDecl>(declRef->getDecl())) {
      localValueTable[varDecl] = newValue;
      return isPrefix ? newValue : oldValue;
    }
  }

  llvm::WithColor::error() << "chwc: unsupported increment/decrement lhs\n";
  return nullptr;
}

} // namespace chwc
