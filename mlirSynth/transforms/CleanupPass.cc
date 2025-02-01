#include "transforms/Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/AnalysisManager.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

using namespace mlir;

namespace mlir {

#define GEN_PASS_DEF_CLEANUP
#include "transforms/Passes.h.inc"

namespace {

struct CleanupPass
    : public impl::CleanupBase<CleanupPass> {
  void runOnOperation() override;
};

}  // namespace
}  // namespace mlir

void CleanupPass::runOnOperation() {
  auto operation = getOperation();

  llvm::SetVector<Operation *> toErase;
  for (auto func : operation.getOps<func::FuncOp>()) {
    if (func->hasAttr("irsynth.original") || func->hasAttr("irsynth.raised"))
      toErase.insert(func);
  }

  for (auto *op : toErase)
    op->erase();
}

std::unique_ptr<OperationPass<ModuleOp>> createCleanupPass() {
  return std::make_unique<CleanupPass>();
}

