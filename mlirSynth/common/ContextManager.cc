#include "ContextManager.h"

#include "Common.h"

#include "gml_st/IR/gml_st_ops.h"
#include "gml_st/transforms/passes.h"
#include "gml_st/transforms/test_passes.h"
#include "lhlo/IR/lhlo_ops.h"
#include "lhlo/transforms/passes.h"
#include "mlir-hlo/Dialect/mhlo/IR/register.h"
#include "mlir-hlo/Dialect/mhlo/transforms/passes.h"
#include "mlir-hlo/Transforms/passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "mlir/Tools/ParseUtilities.h"
#include "stablehlo/dialect/ChloOps.h"
#include "stablehlo/dialect/Register.h"
#include "thlo/IR/thlo_ops.h"
#include "thlo/transforms/passes.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/ThreadPool.h"
#include "llvm/Support/Threading.h"

using namespace mlir;

MLIRContext *ContextManager::createContext() {
  bool parallel = true;

  // Create and initialize.
  MLIRContextPtr ctx = std::make_shared<MLIRContext>();
  if (!parallel) {
    ctx->disableMultithreading();
  }

  if (!printErrors) {
    auto &diagEngine = ctx->getDiagEngine();
    diagEngine.registerHandler([](Diagnostic &diag) {});
  }

  // Register dialects.
  mlir::DialectRegistry registry;
  registerAllDialects(registry);
  registerLLVMDialectTranslation(registry);

  registry.insert<func::FuncDialect>();
  mlir::mhlo::registerAllMhloDialects(registry);
  mlir::stablehlo::registerAllDialects(registry);
  registry.insert<mlir::lmhlo::LmhloDialect, mlir::gml_st::GmlStDialect,
                  mlir::thlo::THLODialect>();

  ctx->appendDialectRegistry(registry);
  ctx->loadAllAvailableDialects();

  ctx->getOrLoadDialect<mhlo::MhloDialect>();
  ctx->getOrLoadDialect<chlo::ChloDialect>();

  LLVM_DEBUG(llvm::dbgs() << "Created new context " << ctx.get() << "\n");

  contexts.push_back(ctx);
  return ctx.get();
}

ContextManager::ContextManager(bool printErrors, bool skipRegistration)
    : printErrors(printErrors) {
  if (skipRegistration) {
    return;
  }

  // Register passes.
  registerAllPasses();
  mlir::hlo::registerLMHLOTransformsPasses();
  mlir::mhlo::registerAllMhloPasses();
  mlir::lmhlo::registerAllLmhloPasses();
  mlir::thlo::registerAllThloPasses();
  mlir::gml_st::registerGmlStPasses();
  mlir::gml_st::registerGmlStTestPasses();
}
