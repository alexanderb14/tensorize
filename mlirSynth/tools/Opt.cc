#include "analysis/PolyhedralAnalysis.h"
#include "transforms/Passes.h"

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

using namespace llvm;
using namespace mlir;

int main(int argc, char **argv) {
  DialectRegistry registry;
  registerAllDialects(registry);
  mlir::mhlo::registerAllMhloDialects(registry);
  mlir::stablehlo::registerAllDialects(registry);

  registerAllPasses();
  mlir::hlo::registerLMHLOTransformsPasses();
  mlir::mhlo::registerAllMhloPasses();
  mlir::lmhlo::registerAllLmhloPasses();
  mlir::thlo::registerAllThloPasses();

  registerPolyhedralAnalysisPass();

  registerChangeSizesPass();
  registerCleanupPass();
  registerCopyModifiedMemrefsPass();
  registerFoldToTensorToMemrefPairPass();
  registerLoopDistributionPass();
  registerLoopOutlinePass();
  registerMemrefToScfPass();
  registerMemrefRank0ToScalarPass();
  registerPrepareTargetPass();
  registerTargetOutlinePass();

  return mlir::asMainReturnCode(mlir::MlirOptMain(
      argc, argv, "Synthesizer opt driver\n", registry, true));
}
