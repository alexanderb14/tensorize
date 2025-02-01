#include "common/Common.h"
#include "common/ContextManager.h"
#include "execution/Executor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Support/LogicalResult.h"
#include "synthesis/Candidate.h"
#include "synthesis/CartesianProduct.h"
#include "synthesis/Guide.h"
#include "synthesis/Synthesizer.h"
#include "transforms/Passes.h"
#include "transforms/Utils.h"

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
#include "mlir/Transforms/Passes.h"
#include "stablehlo/dialect/ChloOps.h"
#include "stablehlo/dialect/Register.h"
#include "thlo/transforms/passes.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/ThreadPool.h"
#include "llvm/Support/Threading.h"

#include <iterator>
#include <stablehlo/dialect/StablehloOps.h>

using namespace llvm;
using namespace mlir;

std::vector<std::string> splitString(std::string &str) {
  std::vector<std::string> vect;
  std::string delimiter = ",";

  std::stringstream ss(str);
  while (ss.good()) {
    std::string substr;
    getline(ss, substr, ',');
    vect.push_back(substr);
  }

  return vect;
}

LogicalResult preprocess(Operation *op, MLIRContext *ctx,
                         SynthesisOptions &options) {
  mlir::PassManager pm(ctx);

  pm.addPass(createAnnotateLastStoredMemrefArgPass());
  if (options.distribute)
    pm.addPass(createLoopDistributionPass());
  pm.addPass(createChangeSizesPass());
  pm.addPass(createLoopOutlinePass());
  pm.addPass(createCopyModifiedMemrefsPass());

  if (failed(pm.run(op))) {
    llvm::errs() << "Failed to run preprocessing passes\n";
    return failure();
  }
  LLVM_DEBUG(llvm::dbgs() << "After preprocessing:\n");
  LLVM_DEBUG(op->print(llvm::dbgs()));

  return success();
}

LogicalResult postprocess(Operation *op, MLIRContext *ctx,
                          SynthesisOptions &options) {
  mlir::PassManager pm(ctx);
  pm.addNestedPass<func::FuncOp>(mhlo::createChloLegalizeToHloPass());
  pm.addPass(mhlo::createHloLegalizeToStablehloPass());
  pm.addPass(createInlinerPass());
  pm.addPass(createFoldToTensorToMemrefPairPass());
  pm.addPass(createCleanupPass());
  // pm.addPass(createTargetOutlinePass());

  if (failed(pm.run(op))) {
    llvm::errs() << "Failed to run postprocessing passes\n";
    return failure();
  }
  LLVM_DEBUG(llvm::dbgs() << "After preprocessing:\n");
  LLVM_DEBUG(op->print(llvm::dbgs()));

  return success();
}

void mergeIntoModule(mlir::MLIRContext *&ctx, OwningOpRef<Operation *> &inputOp,
                     llvm::DenseMap<func::FuncOp, OwningOpRef<ModuleOp>>
                         &originalToSynthesizedFns,
                     llvm::DenseMap<func::FuncOp, std::vector<unsigned int>>
                         &originalToSynthesizedArgIds) {
  OpBuilder builder(ctx);

  for (auto &kv : originalToSynthesizedFns) {
    auto inputFunc = kv.first;
    auto synthesizedFunc = getFunctions(kv.second.get(), "irsynth.raised")[0];
    auto argIds = originalToSynthesizedArgIds[inputFunc];

    synthesizedFunc.setName(
        StringAttr::get(ctx, inputFunc.getName() + "_raised"));

    // Insert the synthesized functions into the original module.
    synthesizedFunc->moveAfter(inputFunc);

    // Get the call sites of the original function.
    std::vector<func::CallOp> callSites;
    inputOp.get()->walk([&](func::CallOp callOp) {
      if (callOp.getCallee() == inputFunc.getName())
        callSites.push_back(callOp);
    });

    // Change the called function to the synthesized function and
    // insert conversion ops for their arguments and results.
    for (auto callSite : callSites) {
      callSite.setCallee(synthesizedFunc.getName());

      // Arguments: Memref to tensor.
      std::vector<mlir::Value> callSiteOperands;
      for (auto operand : callSite.getOperands())
        callSiteOperands.push_back(operand);

      unsigned operandIdx = 0;
      for (operandIdx = 0; operandIdx < argIds.size(); ++operandIdx) {
        unsigned argId = argIds[operandIdx];

        auto callArg = callSiteOperands[argId];
        auto synthesizedArg = synthesizedFunc.getArgument(operandIdx);

        // If types are the same, no need to insert conversion.
        if (callArg.getType() == synthesizedArg.getType())
          continue;

        // If call arg type is a memref, insert conversion from memref to
        // tensor.
        if (callArg.getType().isa<MemRefType>()) {
          // Insert bufferization.to_tensor ops for the call arguments.
          builder.setInsertionPoint(callSite);
          auto toTensorOp = builder.create<bufferization::ToTensorOp>(
              callSite.getLoc(), callArg);

          // Replace the call site argument with the result of the to_tensor op.
          callSite.setOperand(operandIdx, toTensorOp.getResult());

          // If call arg type is a scalar float, insert conversion from float to
          // tensor.
        } else if (callArg.getType().isa<FloatType>()) {
          // Insert tensor.from_elements ops for the call arguments.
          builder.setInsertionPoint(callSite);
          auto tensorType = RankedTensorType::get({}, callArg.getType());
          auto fromElementsOp = builder.create<tensor::FromElementsOp>(
              callSite.getLoc(), tensorType, callArg);

          // Replace the call site argument with the result of the
          // from_elements op.
          callSite.setOperand(operandIdx, fromElementsOp.getResult());
        }
      }
      // Remove the remaining call site arguments.
      callSite->eraseOperands(operandIdx,
                              callSite.getNumOperands() - operandIdx);

      // Results: Tensor to memref.
      assert(callSite.getNumResults() == synthesizedFunc.getNumResults() &&
             "Number of call site results and function results must match");
      for (unsigned resultIdx = 0; resultIdx < callSite.getNumResults();
           resultIdx++) {
        auto callResType = callSite.getResultTypes()[resultIdx];
        auto synthesizedResType = synthesizedFunc.getResultTypes()[resultIdx];

        // If types are the same, no need to insert conversion.
        if (callResType == synthesizedResType)
          continue;

        // If call site type is not a memref, no need to insert conversion.
        if (!callResType.isa<MemRefType>())
          continue;

        // Get call site result.
        auto callRes = callSite.getResult(resultIdx);

        // Insert bufferization.to_memref ops for the call results.
        builder.setInsertionPointAfter(callSite);
        auto toMemrefOp = builder.create<bufferization::ToMemrefOp>(
            UnknownLoc::get(ctx), callResType, callRes);

        // Set the result type of the call site to the type of the synthesized
        // function result.
        callSite.getResult(resultIdx).setType(synthesizedResType);

        // Replace the uses of the call site result with the result of the
        // to_memref op.
        callRes.replaceAllUsesExcept(toMemrefOp.getResult(), toMemrefOp);
      }
    }
  }

  // Get last ToMemrefOp.
  bufferization::ToMemrefOp lastToMemrefOp = nullptr;
  inputOp.get()->walk([&](bufferization::ToMemrefOp toMemrefOp) {
    lastToMemrefOp = toMemrefOp;
  });
  assert(lastToMemrefOp != nullptr && "lastToMemref is null");

  // Copy lastToMemrefOp to the arg memref, the one that has been last stored
  // to, which is annotated with the irSynth.lastStoredMemref attribute.
  // - Get the attribute with the irSynth.lastStoredMemref attribute.
  mlir::Value lastStoredMemrefArg;
  inputOp.get()->walk([&](func::FuncOp func) {
    for (unsigned i = 0; i < func.getNumArguments(); ++i) {
      auto argAttrs = func.getArgAttrDict(i);
      if (argAttrs && argAttrs.contains("irSynth.lastStoredMemref")) {
        lastStoredMemrefArg = func.getArgument(i);
      }
    }
  });
  assert(lastStoredMemrefArg != nullptr && "lastStoredMemrefArg is null");

  // - Create the memref.copy operation.
  builder.setInsertionPointAfter(lastToMemrefOp);
  mlir::Value lastToMemref = lastToMemrefOp.getMemref();
  builder.create<memref::CopyOp>(lastToMemrefOp->getLoc(), lastToMemref,
                                 lastStoredMemrefArg);
}
int main(int argc, char **argv) {
  // Parse command line arguments.
  cl::opt<std::string> inputFilename(cl::Positional, cl::desc("<input file>"),
                                     cl::init("-"));

  cl::opt<std::string> targetDialect(
      "target-dialect", cl::desc("Target dialect"), cl::init("hlo"));

  cl::opt<bool> printStatusNames(
      "print-status-names", cl::desc("Print status names"), cl::init(false));
  cl::opt<bool> printStatusTiles(
      "print-status-tiles", cl::desc("Print status tiles"), cl::init(false));

  cl::opt<bool> printValidCandidates("print-valid-candidates",
                                     cl::desc("Print valid candidates"),
                                     cl::init(false));
  cl::opt<bool> printInvalidCandidates("print-invalid-candidates",
                                       cl::desc("Print invalid candidates"),
                                       cl::init(false));

  cl::opt<bool> printErrors("print-errors", cl::desc("Print errors"),
                            cl::init(false));
  cl::opt<bool> printStats("print-stats", cl::desc("Print stats"),
                           cl::init(false));
  cl::opt<bool> printArgsAndResults("print-args-and-results",
                                    cl::desc("Print args and results"),
                                    cl::init(false));

  cl::opt<bool> printAvailableOps(
      "print-available-ops", cl::desc("Print available ops"), cl::init(false));
  cl::opt<bool> printSynthesisSteps("print-synthesis-steps",
                                    cl::desc("Print synthesis steps"),
                                    cl::init(false));

  cl::opt<std::string> ops(
      "ops", cl::desc("Comma separated list of allowed ops"), cl::init(""));
  cl::opt<int> maxNumOps("max-num-ops", cl::desc("Max number of operations"),
                         cl::init(3));
  cl::opt<int> timeoutPerFunction(
      "timeout-per-function",
      cl::desc("Synthesis timeout per function in seconds"), cl::init(0));

  cl::opt<int> numThreads("num-threads", cl::desc("Number of threads"),
                          cl::init(1));

  cl::opt<bool> ignoreEquivalentCandidates(
      "ignore-equivalent-candidates",
      cl::desc("Ignore computationally equivalent candidates"),
      cl::init(false));
  cl::opt<bool> ignoreTypes(
      "ignore-types",
      cl::desc("Ignore operand types when generating candidates"),
      cl::init(false));
  cl::opt<bool> skipTypeInference(
      "skip-type-inference",
      cl::desc("Skip type inference when generating candidates"),
      cl::init(false));
  cl::opt<bool> skipMergeCandidateArguments(
      "skip-merge-candidate-arguments",
      cl::desc("Skip merging candidate arguments"), cl::init(false));
  cl::opt<bool> withCopyArgs("with-copy-args",
                             cl::desc("Add copy args to candidates"),
                             cl::init(false));

  cl::opt<bool> guide("guide", cl::desc("Use guide to select allowed ops"),
                      cl::init(false));
  cl::opt<bool> distribute(
      "distribute",
      cl::desc("Distribute loops to split synthesis into smaller subproblems"),
      cl::init(false));

  cl::ParseCommandLineOptions(argc, argv, "Synthsizer\n");

  // Parse options.
  SynthesisOptions options;
  options.printStatusNames = printStatusNames;
  options.printStatusTiles = printStatusTiles;
  options.printValidCandidates = printValidCandidates;
  options.printInvalidCandidates = printInvalidCandidates;
  options.printStats = printStats;
  options.printArgsAndResults = printArgsAndResults;
  options.printSynthesisSteps = printSynthesisSteps;
  options.maxNumOps = maxNumOps;
  options.timeoutPerFunction = timeoutPerFunction;
  options.ignoreEquivalentCandidates = ignoreEquivalentCandidates;
  options.ignoreTypes = ignoreTypes;
  options.skipTypeInference = skipTypeInference;
  options.skipMergeCandidateArguments = skipMergeCandidateArguments;
  options.withCopyArgs = withCopyArgs;

  options.guide = guide;
  options.distribute = distribute;

  // Initialize LLVM.
  llvm::InitLLVM y(argc, argv);
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  llvm::InitializeNativeTargetAsmParser();

  // Initialize MLIR.
  ContextManagerPtr contextManager =
      std::make_shared<ContextManager>(printErrors);
  auto *ctx = contextManager->createContext();

  Dialect *stablehloDialect =
      ctx->getOrLoadDialect<stablehlo::StablehloDialect>();
  Dialect *hloDialect = ctx->getOrLoadDialect<mhlo::MhloDialect>();
  Dialect *chloDialect = ctx->getOrLoadDialect<chlo::ChloDialect>();
  Dialect *linalgDialect = ctx->getOrLoadDialect<linalg::LinalgDialect>();

  // Parse the input file.
  std::string errorMessage;
  auto file = openInputFile(inputFilename, &errorMessage);
  if (!file) {
    llvm::errs() << errorMessage << "\n";
    return 1;
  }

  SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(file), SMLoc());

  FallbackAsmResourceMap fallbackResourceMap;
  ParserConfig config(ctx, /*verifyAfterParse=*/true, &fallbackResourceMap);
  OwningOpRef<Operation *> inputModule =
      parseSourceFileForTool(sourceMgr, config, /*insertImplicitModule*/ false);
  assert(inputModule && "Failed to parse input file");
  OwningOpRef<Operation *> inputModuleCpy = inputModule.get()->clone();

  // Preprocess.
  if (preprocess(inputModule.get(), ctx, options).failed()) {
    return 1;
  }

  // Parse the funcion ops.
  std::vector<func::FuncOp> functions =
      getFunctions(inputModule.get(), "irsynth.original");
  LLVM_DEBUG(llvm::dbgs() << "Found " << functions.size()
                          << " functions to synthesize\n");

  IExecutorPtr executor;
  if (numThreads == 1) {
    ctx->disableMultithreading();
    executor = std::make_shared<Executor>(ctx);
  } else {
    executor = std::make_shared<ThreadedExecutor>(contextManager, numThreads);
  }

  // Target dialect specific.
  std::vector<Dialect *> dialects;
  std::vector<std::string> supportedOps;
  InitialCandidateGeneratorPtr initialCandidateGen;

  if (targetDialect == "hlo") {
    dialects = {stablehloDialect, hloDialect, chloDialect};
    supportedOps = {"chlo.broadcast_divide",   "chlo.broadcast_add",
                    "chlo.broadcast_subtract", "chlo.broadcast_multiply",
                    "stablehlo.dot",           "stablehlo.reduce",
                    "stablehlo.dot_general",   "stablehlo.transpose",
                    "stablehlo.select"};
    initialCandidateGen = std::make_shared<HLOInitialCandidateGenerator>(*ctx);
  } else if (targetDialect == "linalg") {
    dialects = {linalgDialect};
    supportedOps = {"linalg.matmul", "linalg.matvec"};
    initialCandidateGen =
        std::make_shared<LinalgInitialCandidateGenerator>(*ctx);

    options.ignoreTypes = true;
    options.skipTypeInference = true;
    options.withCopyArgs = true;
  } else {
    llvm::errs() << "Target dialect not supported\n";
    return 1;
  }

  // Synthesize functions.
  llvm::DenseMap<func::FuncOp, OwningOpRef<ModuleOp>> originalToSynthesizedFns;
  llvm::DenseMap<func::FuncOp, std::vector<unsigned>>
      originalToSynthesizedArgIds;

  bool failedAtLeastOnce = false;

  SynthesisStats stats;
  for (auto inputFuncOrig : functions) {
    auto inputFunc = inputFuncOrig.clone();
    // Get ops.
    std::vector<std::string> opsVec;
    if (guide) {
      opsVec = predictOps(supportedOps, inputFunc);
    } else if (!ops.empty()) {
      opsVec = splitString(ops);
    } else {
      opsVec = supportedOps;
    }
    auto availableOps = getDialectOps(ctx, dialects, opsVec, false);

    // Print available ops
    if (printAvailableOps) {
      llvm::outs() << "Available ops:\n";
      for (auto op : availableOps) {
        llvm::outs() << op << "\n";
      }
    }

    // Synthesize.
    if (options.printSynthesisSteps) {
      llvm::outs() << "Synthesizing function " << inputFunc.getName() << "\n"
                   << "--------------------------\n";
      inputFunc.print(llvm::outs());
    }

    CandidateStorePtr candidateStore = std::make_shared<CandidateStore>();

    SynthesisContextPtr synthCtx = std::make_shared<SynthesisContext>(
        *ctx, executor, initialCandidateGen, candidateStore, options, stats);
    auto result = synthesize(synthCtx, inputFunc, availableOps);

    if (result) {
      if (options.printSynthesisSteps) {
        llvm::errs() << "\033[1;42m"
                     << "Succeeded synthesizing function"
                     << "\033[0m"
                     << "\n";
        result->module->print(llvm::outs());
      }
    } else {
      llvm::errs() << "\033[1;41m"
                   << "Failed synthesizing function"
                   << "\033[0m"
                   << "\n";
      failedAtLeastOnce = true;
    }

    if (result) {
      stats.numOpsPerFunction.push_back(result->candidate->getNumOps());

      originalToSynthesizedFns[inputFuncOrig] = std::move(result->module);
      originalToSynthesizedArgIds[inputFuncOrig] =
          result->candidate->getArgIds();
    }
  }

  if (options.printStats)
    stats.dump();

  mergeIntoModule(ctx, inputModule, originalToSynthesizedFns,
                  originalToSynthesizedArgIds);

  // Print.
  inputModuleCpy.get()->print(llvm::outs());
  inputModule.get()->print(llvm::outs());

  // Postprocess.
  if (postprocess(inputModule.get(), ctx, options).failed()) {
    return 1;
  }

  // Verify.
  if (failed(verify(inputModule.get())))
    return 1;

  if (failedAtLeastOnce)
    return 1;
  return 0;
}
