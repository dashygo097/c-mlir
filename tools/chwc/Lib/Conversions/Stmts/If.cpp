#include "../../Converter.h"
#include "../Utils/Cast.h"
#include "../Utils/Comb.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/WithColor.h"

namespace chwc {

template <typename K>
static void pushUnique(llvm::SmallVectorImpl<K> &keys, K key) {
  for (K existing : keys) {
    if (existing == key) {
      return;
    }
  }

  keys.push_back(key);
}

static auto mergeValue(mlir::OpBuilder &builder, mlir::Location loc,
                       mlir::Value cond, mlir::Value trueValue,
                       mlir::Value falseValue) -> mlir::Value {
  if (!trueValue && !falseValue) {
    return nullptr;
  }

  if (!trueValue) {
    return falseValue;
  }

  if (!falseValue) {
    return trueValue;
  }

  if (trueValue == falseValue) {
    return trueValue;
  }

  if (trueValue.getType() != falseValue.getType()) {
    falseValue =
        utils::promoteValue(builder, loc, falseValue, trueValue.getType());
    if (!falseValue) {
      return nullptr;
    }
  }

  return utils::mux(builder, loc, cond, trueValue, falseValue);
}

template <typename K>
static void mergeMap(mlir::OpBuilder &builder, mlir::Location loc,
                     mlir::Value cond,
                     const llvm::DenseMap<K, mlir::Value> &before,
                     const llvm::DenseMap<K, mlir::Value> &thenMap,
                     const llvm::DenseMap<K, mlir::Value> &elseMap,
                     llvm::DenseMap<K, mlir::Value> &outMap) {
  llvm::SmallVector<K, 32> keys;

  for (auto &it : thenMap) {
    pushUnique(keys, it.first);
  }

  for (auto &it : elseMap) {
    pushUnique(keys, it.first);
  }

  for (K key : keys) {
    mlir::Value beforeValue = before.lookup(key);
    mlir::Value thenValue = thenMap.lookup(key);
    mlir::Value elseValue = elseMap.lookup(key);

    if (!thenValue) {
      thenValue = beforeValue;
    }

    if (!elseValue) {
      elseValue = beforeValue;
    }

    mlir::Value merged = mergeValue(builder, loc, cond, thenValue, elseValue);

    if (merged) {
      outMap[key] = merged;
    }
  }
}

static void mergeLocalMap(
    mlir::OpBuilder &builder, mlir::Location loc, mlir::Value cond,
    const llvm::DenseMap<const clang::VarDecl *, mlir::Value> &before,
    const llvm::DenseMap<const clang::VarDecl *, mlir::Value> &thenMap,
    const llvm::DenseMap<const clang::VarDecl *, mlir::Value> &elseMap,
    llvm::DenseMap<const clang::VarDecl *, mlir::Value> &outMap) {
  llvm::SmallVector<const clang::VarDecl *, 32> keys;

  for (auto &it : thenMap) {
    pushUnique(keys, it.first);
  }

  for (auto &it : elseMap) {
    pushUnique(keys, it.first);
  }

  for (const clang::VarDecl *key : keys) {
    mlir::Value beforeValue = before.lookup(key);

    if (!beforeValue) {
      continue;
    }

    mlir::Value thenValue = thenMap.lookup(key);
    mlir::Value elseValue = elseMap.lookup(key);

    if (!thenValue) {
      thenValue = beforeValue;
    }

    if (!elseValue) {
      elseValue = beforeValue;
    }

    mlir::Value merged = mergeValue(builder, loc, cond, thenValue, elseValue);

    if (merged) {
      outMap[key] = merged;
    }
  }
}

auto CHWConverter::TraverseIfStmt(clang::IfStmt *ifStmt) -> bool {
  if (!ifStmt) {
    return true;
  }

  mlir::OpBuilder &builder = contextManager.Builder();
  mlir::Location loc = builder.getUnknownLoc();

  mlir::Value cond = generateExpr(ifStmt->getCond());
  if (!cond) {
    llvm::WithColor::error() << "chwc: failed to generate if condition\n";
    return true;
  }

  cond = utils::toBool(builder, loc, cond);
  if (!cond) {
    return true;
  }

  auto beforeCurrentValues = moduleContext.currentValues;
  auto beforeNextValues = moduleContext.nextValues;
  auto beforeOutputValues = moduleContext.outputValues;

  llvm::DenseMap<const clang::VarDecl *, mlir::Value> beforeLocals;
  mlir::Value beforeReturnValue = nullptr;
  bool beforeHasReturnValue = false;

  if (!functionStack.empty()) {
    beforeLocals = functionStack.back().locals;
    beforeReturnValue = functionStack.back().returnValue;
    beforeHasReturnValue = functionStack.back().hasReturnValue;
  }

  TraverseStmt(ifStmt->getThen());

  auto thenCurrentValues = moduleContext.currentValues;
  auto thenNextValues = moduleContext.nextValues;
  auto thenOutputValues = moduleContext.outputValues;

  llvm::DenseMap<const clang::VarDecl *, mlir::Value> thenLocals;
  mlir::Value thenReturnValue = nullptr;
  bool thenHasReturnValue = false;

  if (!functionStack.empty()) {
    thenLocals = functionStack.back().locals;
    thenReturnValue = functionStack.back().returnValue;
    thenHasReturnValue = functionStack.back().hasReturnValue;
  }

  moduleContext.currentValues = beforeCurrentValues;
  moduleContext.nextValues = beforeNextValues;
  moduleContext.outputValues = beforeOutputValues;

  if (!functionStack.empty()) {
    functionStack.back().locals = beforeLocals;
    functionStack.back().returnValue = beforeReturnValue;
    functionStack.back().hasReturnValue = beforeHasReturnValue;
  }

  if (clang::Stmt *elseStmt = ifStmt->getElse()) {
    TraverseStmt(elseStmt);
  }

  auto elseCurrentValues = moduleContext.currentValues;
  auto elseNextValues = moduleContext.nextValues;
  auto elseOutputValues = moduleContext.outputValues;

  llvm::DenseMap<const clang::VarDecl *, mlir::Value> elseLocals;
  mlir::Value elseReturnValue = nullptr;
  bool elseHasReturnValue = false;

  if (!functionStack.empty()) {
    elseLocals = functionStack.back().locals;
    elseReturnValue = functionStack.back().returnValue;
    elseHasReturnValue = functionStack.back().hasReturnValue;
  }

  moduleContext.currentValues = beforeCurrentValues;
  moduleContext.nextValues = beforeNextValues;
  moduleContext.outputValues = beforeOutputValues;

  mergeMap(builder, loc, cond, beforeCurrentValues, thenCurrentValues,
           elseCurrentValues, moduleContext.currentValues);

  mergeMap(builder, loc, cond, beforeNextValues, thenNextValues, elseNextValues,
           moduleContext.nextValues);

  mergeMap(builder, loc, cond, beforeOutputValues, thenOutputValues,
           elseOutputValues, moduleContext.outputValues);

  if (!functionStack.empty()) {
    functionStack.back().locals = beforeLocals;
    functionStack.back().returnValue = beforeReturnValue;
    functionStack.back().hasReturnValue = beforeHasReturnValue;

    mergeLocalMap(builder, loc, cond, beforeLocals, thenLocals, elseLocals,
                  functionStack.back().locals);

    if (thenHasReturnValue && elseHasReturnValue) {
      mlir::Value mergedReturn =
          mergeValue(builder, loc, cond, thenReturnValue, elseReturnValue);

      if (mergedReturn) {
        functionStack.back().returnValue = mergedReturn;
        functionStack.back().hasReturnValue = true;
      }
    }
  }

  return true;
}

} // namespace chwc
