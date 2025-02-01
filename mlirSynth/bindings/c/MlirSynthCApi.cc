#include "bindings/c/MlirSynthCApi.h"

#include "execution/Executor.h"
#include "execution/Lowering.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "synthesis/CandidateStore.h"
#include "synthesis/Generators.h"
#include "synthesis/Guide.h"
#include "synthesis/Synthesizer.h"
#include "transforms/Utils.h"
#include "translation/TranslateToPython.h"

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"

#include "mlir/CAPI/IR.h"
#include "mlir/CAPI/Support.h"
#include "mlir/CAPI/Utils.h"

#include "mlir-hlo/Dialect/mhlo/IR/register.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Verifier.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"

#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "stablehlo/dialect/ChloOps.h"
#include "stablehlo/dialect/Register.h"
#include <stablehlo/dialect/StablehloOps.h>

#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/TargetSelect.h"

#include <cstddef>

using namespace mlir;

void registerDialects(MlirContext context) {
  auto ctx = unwrap(context);

  DialectRegistry registry;
  registerAllDialects(registry);
  registerLLVMDialectTranslation(registry);
  mlir::mhlo::registerAllMhloDialects(registry);
  mlir::stablehlo::registerAllDialects(registry);
  ctx->appendDialectRegistry(registry);
  ctx->loadAllAvailableDialects();
}

void lowerCHLOToAffine(MlirModule module, bool expandArithOps) {
  auto mod = unwrap(module);

  auto pm = std::make_shared<PassManager>(mod->getContext());
  HLO::addCHLOToAffinePasses(pm, expandArithOps);
  pm->run(mod);
}

void emitPython(MlirModule module, StdString pyCode) {
  auto mod = unwrap(module);

  auto pc = unwrap(pyCode);

  std::string buffer;
  llvm::raw_string_ostream llvmStream{buffer};

  if (failed(translateToPython(mod, llvmStream, false))) {
    buffer = "";
  }

  pc->assign(buffer);
}

void predictGuideOps(MlirOperation op, StdString ops) {
  auto opp = unwrap(op);

  auto pc = unwrap(ops);

  std::string buffer;

  std::vector<std::string> unused;
  auto guideOps = predictOps(unused, opp);

  for (auto [opIdx, op] : llvm::enumerate(guideOps)) {
    buffer += op;
    if (opIdx < guideOps.size() - 1)
      buffer += ",";
  }

  pc->assign(buffer);
}

void predictConstants(MlirOperation op, StdString constants) {
  auto opp = unwrap(op);

  auto pc = unwrap(constants);

  std::vector<float> consts = predictConstants(opp);

  for (auto &constant : consts) {
    pc->append(std::to_string(constant));
    if (&constant != &consts.back())
      pc->append(",");
  }
}

void enumerateOneOp(MlirStringRef *operationsWrapped, unsigned operationsSize,
                    MlirType *argTypesWrapped, unsigned argTypesSize,
                    MlirType returnTypeWrapped, SynthesisOptions options,
                    void *resultModuleStrs) {
  // Unwrap args.
  SmallVector<StringRef, 4> opsUnwrappedStore;
  auto opsUnwrapped =
      unwrapList(operationsSize, operationsWrapped, opsUnwrappedStore);
  std::vector<std::string> ops;
  for (auto &op : opsUnwrapped)
    ops.push_back(op.str());

  std::vector<mlir::Type> argTypes;
  for (unsigned i = 0; i < argTypesSize; i++)
    argTypes.push_back(unwrap(argTypesWrapped[i]));

  mlir::Type returnType = unwrap(returnTypeWrapped);

  // Initialize LLVM.
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  llvm::InitializeNativeTargetAsmParser();

  // ContextManager
  bool printErrors = false;
  ContextManagerPtr contextManager =
      std::make_shared<ContextManager>(printErrors, true);
  auto *ctx = contextManager->createContext();

  // Executor
  IExecutorPtr executor;
  int numThreads = 32;
  if (numThreads == 1) {
    ctx->disableMultithreading();
    executor = std::make_shared<Executor>(ctx);
  } else {
    executor = std::make_shared<ThreadedExecutor>(contextManager, numThreads);
  }

  // Initial candidate generator.
  auto initialCandidateGen =
      std::make_shared<HLOInitialCandidateGenerator>(*ctx);
  Dialect *stablehloDialect =
      ctx->getOrLoadDialect<stablehlo::StablehloDialect>();
  Dialect *hloDialect = ctx->getOrLoadDialect<mhlo::MhloDialect>();
  Dialect *chloDialect = ctx->getOrLoadDialect<chlo::ChloDialect>();
  std::vector<Dialect *> dialects = {stablehloDialect, hloDialect, chloDialect};
  auto availableOps = getDialectOps(ctx, dialects, ops, false);

  // Candidate store.
  CandidateStorePtr candidateStore = std::make_shared<CandidateStore>();

  // Stats.
  SynthesisStats stats;

  SynthesisContextPtr synthCtx = std::make_shared<SynthesisContext>(
      *ctx, executor, initialCandidateGen, candidateStore, options, stats);
  enumerateOneOp(synthCtx, argTypes, returnType, availableOps);

  // Convert all candidate modules to strings.
  auto *resultModulesStrVec =
      (llvm::SmallVector<std::string> *)resultModuleStrs;

  auto candidates = candidateStore->getCandidates();
  for (auto &candidate : candidates) {
    auto module = createModule(*ctx, candidate);

    std::string buffer;
    llvm::raw_string_ostream llvmStream{buffer};
    module->print(llvmStream);

    resultModulesStrVec->push_back(buffer);
  }

  // stats.dump();
}