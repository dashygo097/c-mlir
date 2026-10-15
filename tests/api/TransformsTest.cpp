#include "cmlir/Transforms/Passes.h"
#include "cmlir/Transforms/Pipeline.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/ADT/APInt.h"
#include "llvm/Support/raw_ostream.h"

namespace {

auto check(bool condition, llvm::StringRef message) -> bool {
  if (!condition) {
    llvm::errs() << "FAIL: " << message << '\n';
  }
  return condition;
}

auto checkCastPasses(mlir::MLIRContext &ir) -> bool {
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
    module {
      func.func @lossy(%value: i32) -> i32 {
        %float = arith.sitofp %value : i32 to f32
        %result = arith.fptosi %float : f32 to i32
        return %result : i32
      }
      func.func @exact(%value: i32) -> i32 {
        %float = arith.sitofp %value : i32 to f64
        %result = arith.fptosi %float : f64 to i32
        return %result : i32
      }
      func.func @unsigned_exact(%value: i16) -> i16 {
        %float = arith.uitofp %value : i16 to f32
        %result = arith.fptoui %float : f32 to i16
        return %result : i16
      }
      func.func @wide_constant() -> i128 {
        %integer = arith.constant 1267650600228229401496703205377 : i128
        %float = arith.uitofp %integer : i128 to f128
        %result = arith.fptoui %float : f128 to i128
        return %result : i128
      }
      func.func @lossy_index(%value: index) -> index {
        %integer = arith.index_cast %value : index to i32
        %result = arith.index_cast %integer : i32 to index
        return %result : index
      }
      func.func @exact_index(%value: index) -> index {
        %integer = arith.index_cast %value : index to i64
        %result = arith.index_cast %integer : i64 to index
        return %result : index
      }
      func.func @bounded_index(%pointer: memref<?xi32>) {
        %zero = arith.constant 0 : index
        %one = arith.constant 1 : index
        %bound = arith.constant 8 : index
        scf.for %iv = %zero to %bound step %one {
          %integer = arith.index_cast %iv : index to i32
          %index = arith.index_cast %integer : i32 to index
          %value = arith.constant 1 : i32
          memref.store %value, %pointer[%index] : memref<?xi32>
        }
        return
      }
    }
  )mlir",
                                                        &ir);
  if (!check(bool(module), "cast regression IR parses")) {
    return false;
  }
  mlir::PassManager pm(&ir);
  pm.addPass(cmlir::createArithCastPropPass());
  pm.addPass(cmlir::createConstPropPass());
  if (!check(mlir::succeeded(pm.run(*module)), "standalone cast passes run")) {
    return false;
  }
  auto lossy = module->lookupSymbol<mlir::func::FuncOp>("lossy");
  auto lossyReturn = mlir::cast<mlir::func::ReturnOp>(lossy.front().back());
  if (!check(
          bool(
              lossyReturn.getOperand(0).getDefiningOp<mlir::arith::FPToSIOp>()),
          "lossy i32/f32 round trip remains")) {
    return false;
  }
  for (auto name : {"exact", "unsigned_exact", "exact_index"}) {
    auto function = module->lookupSymbol<mlir::func::FuncOp>(name);
    auto result = mlir::cast<mlir::func::ReturnOp>(function.front().back());
    if (!check(result.getOperand(0) == function.getArgument(0),
               "proven exact round trip is eliminated")) {
      return false;
    }
  }
  auto indexFunction = module->lookupSymbol<mlir::func::FuncOp>("lossy_index");
  auto indexReturn =
      mlir::cast<mlir::func::ReturnOp>(indexFunction.front().back());
  if (!check(bool(indexReturn.getOperand(0)
                      .getDefiningOp<mlir::arith::IndexCastOp>()),
             "unbounded index/i32 round trip remains")) {
    return false;
  }
  auto bounded = module->lookupSymbol<mlir::func::FuncOp>("bounded_index");
  bool hasIndexCast = false;
  bounded.walk([&](mlir::arith::IndexCastOp) -> void { hasIndexCast = true; });
  if (!check(!hasIndexCast, "range-proven induction casts are eliminated")) {
    return false;
  }
  auto wide = module->lookupSymbol<mlir::func::FuncOp>("wide_constant");
  auto wideReturn = mlir::cast<mlir::func::ReturnOp>(wide.front().back());
  auto constant =
      wideReturn.getOperand(0).getDefiningOp<mlir::arith::ConstantOp>();
  return check(
      constant &&
          mlir::cast<mlir::IntegerAttr>(constant.getValue()).getValue() ==
              llvm::APInt(128, "1267650600228229401496703205377", 10),
      "i128/f128 constant folding preserves bits beyond host widths");
}

auto checkAggregatePromotion(mlir::MLIRContext &ir) -> bool {
  auto module = mlir::parseSourceString<mlir::ModuleOp>(R"mlir(
    module {
      func.func @aggregate(%flag: i1, %x: i32, %y: i32) -> i32 {
        %zero = arith.constant 0 : index
        %one = arith.constant 1 : index
        %slots = memref.alloca() : memref<2xi32>
        cf.cond_br %flag, ^first, ^second
      ^first:
        memref.store %x, %slots[%zero] : memref<2xi32>
        memref.store %y, %slots[%one] : memref<2xi32>
        cf.br ^join
      ^second:
        memref.store %y, %slots[%zero] : memref<2xi32>
        memref.store %x, %slots[%one] : memref<2xi32>
        cf.br ^join
      ^join:
        %a = memref.load %slots[%zero] : memref<2xi32>
        %b = memref.load %slots[%one] : memref<2xi32>
        %difference = arith.subi %a, %b : i32
        return %difference : i32
      }
    }
  )mlir",
                                                        &ir);
  if (!check(bool(module), "aggregate regression IR parses")) {
    return false;
  }
  if (auto error = cmlir::runPipeline(*module)) {
    llvm::errs() << llvm::toString(std::move(error)) << '\n';
    return false;
  }
  bool hasStorage = false;
  module->walk([&](mlir::Operation *op) -> void {
    hasStorage |= mlir::isa<mlir::memref::AllocaOp, mlir::memref::LoadOp,
                            mlir::memref::StoreOp>(op);
  });
  return check(!hasStorage && mlir::succeeded(mlir::verify(*module)),
               "SROA and mem2reg promote branch-dependent aggregate fields");
}

} // namespace

auto main() -> int {
  cmlir::registerTransformsPasses();
  mlir::DialectRegistry registry;
  registry.insert<mlir::arith::ArithDialect, mlir::func::FuncDialect,
                  mlir::cf::ControlFlowDialect, mlir::memref::MemRefDialect,
                  mlir::scf::SCFDialect>();
  mlir::MLIRContext ir(registry);
  ir.loadAllAvailableDialects();
  if (!checkCastPasses(ir) || !checkAggregatePromotion(ir)) {
    return 1;
  }
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
