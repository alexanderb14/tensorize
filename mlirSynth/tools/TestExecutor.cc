#include "analysis/PolyhedralAnalysis.h"
#include "execution/ArgUtils.h"
#include "execution/ArrayUtils.h"
#include "execution/Executor.h"
#include "execution/Lowering.h"
#include "validation/CheckingValidator.h"
#include "validation/TestingValidator.h"
#include "synthesis/Synthesizer.h"
#include "transforms/Utils.h"
#include "transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Tools/ParseUtilities.h"
#include "lhlo/transforms/passes.h"
#include "mlir-hlo/Dialect/mhlo/IR/register.h"
#include "mlir-hlo/Dialect/mhlo/transforms/passes.h"
#include "mlir-hlo/Transforms/passes.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "stablehlo/dialect/Register.h"
#include "thlo/transforms/passes.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"

using namespace llvm;
using namespace mlir;

OwningOpRef<ModuleOp> lowerHLO(func::FuncOp &func) {
  auto *ctx = func->getContext();

  auto pm = std::make_shared<mlir::PassManager>(ctx);
  HLO::addCHLOToAffinePasses(pm);

  auto hloModule = createModule(*ctx, &func).release();

  if (failed(pm->run(hloModule))) {
    assert(false && "Couldn't lower to HLO to affine dialect");
  }

  return hloModule;
}

void printStatus(bool isOk, bool printOutputBlackWhite) {
  if (printOutputBlackWhite) {
    if (isOk) {
      llvm::outs() << "OK";
    } else {
      llvm::outs() << "FAIL";
    }
  } else {
    if (isOk) {
      llvm::outs() << "\033[1;42m\033[1;30mOK\033[0m";
    } else {
      llvm::outs() << "\033[1;41m\033[1;30mFAIL\033[0m";
    }
  }
}

int main(int argc, char **argv) {
  // Parse command line arguments.
  cl::opt<std::string> inputFilename(cl::Positional, cl::desc("<input file>"),
                                     cl::init("-"));
  cl::opt<bool> printArgsAndResults("print-args-and-results",
                                    cl::desc("Print args and results"),
                                    cl::init(false));
  cl::opt<bool> printResults("print-results", cl::desc("Print results"),
                             cl::init(false));
  cl::opt<bool> printOutputBlackWhite("print-output-black-white",
                                      cl::desc("Print output black/white"),
                                      cl::init(false));
  cl::ParseCommandLineOptions(argc, argv, "Test Executor\n");

  // Initialize LLVM.
  llvm::InitLLVM y(argc, argv);
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  llvm::InitializeNativeTargetAsmParser();

  // Register dialects and passes.
  DialectRegistry registry;
  registerAllDialects(registry);
  registerLLVMDialectTranslation(registry);
  mlir::mhlo::registerAllMhloDialects(registry);
  mlir::stablehlo::registerAllDialects(registry);

  registerAllPasses();
  mlir::hlo::registerLMHLOTransformsPasses();
  mlir::mhlo::registerAllMhloPasses();
  mlir::lmhlo::registerAllLmhloPasses();
  mlir::thlo::registerAllThloPasses();

  // Create a context.
  MLIRContext ctx;
  ctx.appendDialectRegistry(registry);
  ctx.loadAllAvailableDialects();

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
  ParserConfig config(&ctx, /*verifyAfterParse=*/true, &fallbackResourceMap);
  OwningOpRef<Operation *> inputOp =
      parseSourceFileForTool(sourceMgr, config, /*insertImplicitModule*/ false);
  assert(inputOp && "Failed to parse input file");

  // Load original function module (in affine).
  auto originalFunctions = getFunctions(inputOp.get(), "irsynth.original");
  assert(originalFunctions.size() == 1 &&
         "Expected one function with the irsynth.original attribute");
  auto original = createModule(ctx, &originalFunctions[0]);

  // Load HLO function module(s).
  auto hloFunctions = getFunctions(inputOp.get(), "irsynth.raised");
  for (auto hloFunction : hloFunctions) {
    auto lowered = lowerHLO(hloFunction);

    // Test validate.
    bool testEquiv = testValidate(original->clone(), lowered->clone(),
                                  printArgsAndResults, printResults);
    llvm::outs() << hloFunction.getName().str() << ": Testing with IO ";
    printStatus(testEquiv, printOutputBlackWhite);
    llvm::outs() << "\n";

    // Check validate.
    bool checkEquiv = checkValidate(original->clone(), lowered->clone(),
                                    printArgsAndResults, printResults);
    llvm::outs() << hloFunction.getName().str() << ": Checking with CBMC ";
    printStatus(checkEquiv, printOutputBlackWhite);
    llvm::outs() << "\n";
  }
}
