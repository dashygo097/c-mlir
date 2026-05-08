#include "../../Converter.h"
#include "../Utils/Constant.h"
#include "llvm/Support/WithColor.h"

namespace chwc {

struct StaticForInfo {
  const clang::VarDecl *varDecl{nullptr};
  int64_t begin{0};
  int64_t end{0};
  int64_t step{1};
};

auto getForConstantInt(clang::Expr *expr) -> std::optional<int64_t> {
  if (!expr) {
    return std::nullopt;
  }

  expr = expr->IgnoreParenImpCasts();

  if (auto *lit = mlir::dyn_cast<clang::IntegerLiteral>(expr)) {
    return lit->getValue().getSExtValue();
  }

  return std::nullopt;
}

auto getForDeclRefVar(clang::Expr *expr) -> const clang::VarDecl * {
  if (!expr) {
    return nullptr;
  }

  expr = expr->IgnoreParenImpCasts();

  if (auto *declRef = mlir::dyn_cast<clang::DeclRefExpr>(expr)) {
    return mlir::dyn_cast<clang::VarDecl>(declRef->getDecl());
  }

  return nullptr;
}

auto analyzeStaticFor(clang::ForStmt *forStmt) -> std::optional<StaticForInfo> {
  auto *declStmt = mlir::dyn_cast_or_null<clang::DeclStmt>(forStmt->getInit());
  if (!declStmt || !declStmt->isSingleDecl()) {
    return std::nullopt;
  }

  auto *varDecl = mlir::dyn_cast<clang::VarDecl>(declStmt->getSingleDecl());
  if (!varDecl || !varDecl->hasInit()) {
    return std::nullopt;
  }

  std::optional<int64_t> begin = getForConstantInt(varDecl->getInit());
  if (!begin) {
    return std::nullopt;
  }

  auto *cond =
      mlir::dyn_cast_or_null<clang::BinaryOperator>(forStmt->getCond());
  if (!cond) {
    return std::nullopt;
  }

  const clang::VarDecl *condVar = getForDeclRefVar(cond->getLHS());
  if (!condVar || condVar->getCanonicalDecl() != varDecl->getCanonicalDecl()) {
    return std::nullopt;
  }

  std::optional<int64_t> rawEnd = getForConstantInt(cond->getRHS());
  if (!rawEnd) {
    return std::nullopt;
  }

  auto *inc = mlir::dyn_cast_or_null<clang::UnaryOperator>(forStmt->getInc());
  if (!inc) {
    return std::nullopt;
  }

  const clang::VarDecl *incVar = getForDeclRefVar(inc->getSubExpr());
  if (!incVar || incVar->getCanonicalDecl() != varDecl->getCanonicalDecl()) {
    return std::nullopt;
  }

  int64_t step = 0;

  if (inc->getOpcode() == clang::UO_PreInc ||
      inc->getOpcode() == clang::UO_PostInc) {
    step = 1;
  } else if (inc->getOpcode() == clang::UO_PreDec ||
             inc->getOpcode() == clang::UO_PostDec) {
    step = -1;
  } else {
    return std::nullopt;
  }

  StaticForInfo info;
  info.varDecl = varDecl;
  info.begin = *begin;
  info.step = step;

  switch (cond->getOpcode()) {
  case clang::BO_LT:
    info.end = *rawEnd;
    return step > 0 ? std::optional<StaticForInfo>(info) : std::nullopt;

  case clang::BO_LE:
    info.end = *rawEnd + 1;
    return step > 0 ? std::optional<StaticForInfo>(info) : std::nullopt;

  case clang::BO_GT:
    info.end = *rawEnd;
    return step < 0 ? std::optional<StaticForInfo>(info) : std::nullopt;

  case clang::BO_GE:
    info.end = *rawEnd - 1;
    return step < 0 ? std::optional<StaticForInfo>(info) : std::nullopt;

  default:
    return std::nullopt;
  }
}

auto CHWConverter::TraverseForStmt(clang::ForStmt *forStmt) -> bool {
  std::optional<StaticForInfo> info = analyzeStaticFor(forStmt);
  if (!info) {
    llvm::WithColor::error()
        << "chwc: only statically-bounded for loops are supported for now\n";
    return true;
  }

  if (functionStack.empty()) {
    functionStack.emplace_back();
  }

  HWFunctionContext &frame = functionStack.back();

  mlir::OpBuilder &builder = contextManager.Builder();
  mlir::Location loc = builder.getUnknownLoc();

  mlir::Type ivType = builder.getIntegerType(32);

  if (info->step > 0) {
    for (int64_t i = info->begin; i < info->end; i += info->step) {
      frame.locals[info->varDecl] = utils::intConst(builder, loc, ivType, i);
      TraverseStmt(forStmt->getBody());
    }
  } else {
    for (int64_t i = info->begin; i > info->end; i += info->step) {
      frame.locals[info->varDecl] = utils::intConst(builder, loc, ivType, i);
      TraverseStmt(forStmt->getBody());
    }
  }

  frame.locals.erase(info->varDecl);
  return true;
}

} // namespace chwc
