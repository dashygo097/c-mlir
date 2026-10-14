#ifndef CMLIR_FRONTEND_CALL_LOWERING_H
#define CMLIR_FRONTEND_CALL_LOWERING_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"
#include "clang/AST/Decl.h"
#include <string>

namespace clang_mlir {
class Context;
} // namespace clang_mlir

namespace cmlir::frontend {

class TypeLowering;

struct CallArg {
  mlir::Value value;
  clang::QualType clangType;
};

// Internal pointers use the frontend's memref representation. External
// signatures describe the platform ABI, independently of the call operands.
enum class CallConvention { Internal, External };

auto callConvention(const clang::FunctionDecl &) -> CallConvention;
auto lowerFunctionSignature(TypeLowering &, const clang::FunctionDecl &,
                            CallConvention)
    -> mlir::FailureOr<mlir::FunctionType>;
auto functionSymbolName(const clang::FunctionDecl &) -> std::string;
auto getOrCreateFunctionDecl(mlir::OpBuilder &, mlir::ModuleOp, llvm::StringRef,
                             mlir::FunctionType) -> mlir::func::FuncOp;
auto prepareCallArgument(mlir::OpBuilder &, mlir::Location, const CallArg &,
                         mlir::Type) -> mlir::Value;

class CallLowering {
public:
  CallLowering(clang_mlir::Context &context, TypeLowering &types)
      : context(context), types(types) {}

  // A successful void call contains an empty Value; failure is distinct.
  auto emit(const clang::FunctionDecl &, clang::SourceLocation,
            llvm::ArrayRef<CallArg>) -> mlir::FailureOr<mlir::Value>;

private:
  clang_mlir::Context &context;
  TypeLowering &types;
};

} // namespace cmlir::frontend

#endif // CMLIR_FRONTEND_CALL_LOWERING_H
