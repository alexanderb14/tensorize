#include "mlir/Pass/PassManager.h"
#include "translation/TranslateToPython.h"

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/InitAllDialects.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Tools/ParseUtilities.h"
#include "llvm/Support/SourceMgr.h"

using namespace llvm;
using namespace mlir;

int main(int argc, char **argv) {
  // Parse command line arguments.
  cl::opt<std::string> inputFilename(cl::Positional, cl::desc("<input file>"),
                                     cl::init("-"));
  cl::ParseCommandLineOptions(argc, argv, "Test Executor\n");

  DialectRegistry registry;
  registerAllDialects(registry);

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

  mlir::PassManager pm(&ctx);
  pm.addPass(createLowerAffinePass());
  if (failed(pm.run(*inputOp)))
    return 1;

  if (failed(translateToPython(*inputOp, llvm::outs(), false)))
    return 1;
}