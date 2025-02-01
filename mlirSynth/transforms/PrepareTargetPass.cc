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

#define GEN_PASS_DEF_PREPARETARGET
#include "transforms/Passes.h.inc"

namespace {

struct PrepareTargetPass
    : public impl::PrepareTargetBase<PrepareTargetPass> {
  void runOnOperation() override;
};

}  // namespace
}  // namespace mlir

void PrepareTargetPass::runOnOperation() {
  auto module = getOperation();

  llvm::SetVector<Operation *> toKeep, toErase;
  for (auto func : module.getOps<func::FuncOp>()) {
    if (func->hasAttr("irsynth.target")) {
      toKeep.insert(func);
      func->removeAttr("irsynth.target");
    } else {
      toErase.insert(func);
    }
  }
  assert(toKeep.size() == 1 &&
         "Expected exactly one function with irsynth.target attribute");

  // Rename the target function to "main".
  auto targetFunc = cast<func::FuncOp>(toKeep.front());
  targetFunc->setAttr("sym_name",
                      StringAttr::get(targetFunc->getContext(), "main"));

  // Remove all other functions.
  for (auto *op : toErase)
    op->erase();

  // Strip attributes from the module.
  for (auto attr : module->getAttrs())
    module->removeAttr(attr.getName());
}

std::unique_ptr<OperationPass<ModuleOp>> createPrepareTargetPass() {
  return std::make_unique<PrepareTargetPass>();
}
