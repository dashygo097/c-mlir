#include "../Types/TypeLowering.h"
#include "CallLowering.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "clang/AST/DeclCXX.h"
#include "clang/Basic/SourceManager.h"

namespace cmlir::frontend {

auto callConvention(const clang::FunctionDecl &decl) -> CallConvention {
  const auto *definition = decl.getDefinition();
  if (!definition || decl.getASTContext().getSourceManager().isInSystemHeader(
                         definition->getLocation())) {
    return CallConvention::External;
  }
  return CallConvention::Internal;
}

auto lowerFunctionSignature(TypeLowering &types,
                            const clang::FunctionDecl &decl,
                            CallConvention convention)
    -> mlir::FailureOr<mlir::FunctionType> {
  bool nativeABI =
      convention == CallConvention::External ||
      (decl.getASTContext().getLangOpts().CPlusPlus && decl.isExternC());
  auto convert = [&](clang::QualType type) -> mlir::Type {
    if (type->isReferenceType()) {
      auto address = types.convertType(type);
      if (!address) {
        return {};
      }
      if (nativeABI) {
        return mlir::LLVM::LLVMPointerType::get(address.getContext());
      }
      return address;
    }
    if (!nativeABI) {
      return types.convertType(type);
    }
    // Aggregate ABI classification needs target-specific handling. Reject it
    // before constructing a signature instead of passing an LLVM struct by
    // value and hoping the platform ABI agrees.
    if (!type->isVoidType() && !type->isIntegerType() &&
        !type->isEnumeralType() && !type->isRealFloatingType() &&
        !type->isPointerType()) {
      return {};
    }
    return types.convertStorageType(type);
  };

  llvm::SmallVector<mlir::Type, 4> inputs;
  if (auto *method = llvm::dyn_cast<clang::CXXMethodDecl>(&decl)) {
    if (!method->isStatic()) {
      auto type = convert(method->getThisType());
      if (!type) {
        return mlir::failure();
      }
      inputs.push_back(type);
    }
  }
  for (const auto *parameter : decl.parameters()) {
    auto type = convert(parameter->getType());
    if (!type) {
      return mlir::failure();
    }
    inputs.push_back(type);
  }
  auto result = decl.getReturnType()->isPointerType()
                    ? types.convertStorageType(decl.getReturnType())
                    : convert(decl.getReturnType());
  if (!result) {
    return mlir::failure();
  }
  llvm::SmallVector<mlir::Type, 1> results;
  if (!mlir::isa<mlir::NoneType>(result)) {
    results.push_back(result);
  }
  return mlir::FunctionType::get(result.getContext(), inputs, results);
}

} // namespace cmlir::frontend
