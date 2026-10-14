#include "cmlir/Transforms/Passes.h"
#include "cmlir/Transforms/Pipeline.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Verifier.h"
#include "llvm/Support/raw_ostream.h"

namespace {

auto check(bool condition, llvm::StringRef message) -> bool {
  if (!condition) {
    llvm::errs() << "FAIL: " << message << '\n';
  }
  return condition;
}

} // namespace

auto main() -> int {
  cmlir::registerTransformsPasses();
  mlir::DialectRegistry registry;
  registry.insert<mlir::arith::ArithDialect, mlir::func::FuncDialect>();
  mlir::MLIRContext ir(registry);
  ir.loadAllAvailableDialects();
  mlir::OpBuilder builder(&ir);
  auto loc = builder.getUnknownLoc();
  mlir::OwningOpRef<mlir::ModuleOp> module(mlir::ModuleOp::create(loc));
  auto signature = builder.getFunctionType({}, {builder.getI32Type()});
  builder.setInsertionPointToStart(module->getBody());
  auto helper = mlir::func::FuncOp::create(builder, loc, "helper", signature);
  helper.setPrivate();
  builder.setInsertionPointToStart(helper.addEntryBlock());
  auto one = mlir::arith::ConstantIntOp::create(builder, loc, 1, 32);
  mlir::func::ReturnOp::create(builder, loc, mlir::ValueRange{one});
  builder.setInsertionPointToEnd(module->getBody());
  auto entry = mlir::func::FuncOp::create(builder, loc, "entry", signature);
  builder.setInsertionPointToStart(entry.addEntryBlock());
  auto call =
      mlir::func::CallOp::create(builder, loc, helper, mlir::ValueRange{});
  auto two = mlir::arith::ConstantIntOp::create(builder, loc, 2, 32);
  auto sum = mlir::arith::AddIOp::create(builder, loc, call.getResult(0), two);
  mlir::func::ReturnOp::create(builder, loc, mlir::ValueRange{sum});

  cmlir::PipelineOptions options;
  options.inlineFunctions = true;
  if (auto error = cmlir::runPipeline(module.get(), options)) {
    llvm::errs() << llvm::toString(std::move(error)) << '\n';
    return 1;
  }
  if (!check(mlir::succeeded(mlir::verify(module.get())),
             "optimized module verifies")) {
    return 1;
  }
  auto result = mlir::cast<mlir::func::ReturnOp>(entry.getBody().back().back());
  auto constant = result.getOperand(0).getDefiningOp<mlir::arith::ConstantOp>();
  if (!check(constant &&
                 mlir::cast<mlir::IntegerAttr>(constant.getValue()).getInt() ==
                     3,
             "standalone pipeline inlines and folds the result")) {
    return 1;
  }

  builder.setInsertionPointToEnd(module->getBody());
  auto broken = mlir::func::FuncOp::create(builder, loc, "broken", signature);
  builder.setInsertionPointToStart(broken.addEntryBlock());
  mlir::func::ReturnOp::create(builder, loc);
  bool diagnosed = false;
  mlir::ScopedDiagnosticHandler handler(
      &ir, [&](mlir::Diagnostic &) -> mlir::LogicalResult {
        diagnosed = true;
        return mlir::success();
      });
  options.optimize = false;
  auto error = cmlir::runPipeline(module.get(), options);
  if (!check(bool(error) && diagnosed, "disabled pipeline still verifies")) {
    return 1;
  }
  llvm::consumeError(std::move(error));
  llvm::outs() << "CMLIRTransforms API checks passed\n";
  return 0;
}
