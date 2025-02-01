#include "transforms/Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/AnalysisManager.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

using namespace mlir;

namespace mlir {

#define GEN_PASS_DEF_ANNOTATELASTSTOREDMEMREFARG
#include "transforms/Passes.h.inc"

namespace {

struct AnnotateLastStoredMemrefArgPass
    : public impl::AnnotateLastStoredMemrefArgBase<AnnotateLastStoredMemrefArgPass> {
  void runOnOperation() override;
};

}  // namespace
}  // namespace mlir

void AnnotateLastStoredMemrefArgPass::runOnOperation() {
  auto op = getOperation();

  op->walk([&](func::FuncOp func) {
    // Identify the argument that has been last stored to by the function.
    Value lastStore;
    func.walk([&](AffineStoreOp store) { lastStore = store.getMemRef(); });

    // Annotate the output argument.
    if (auto blockArg = llvm::dyn_cast<BlockArgument>(lastStore)) {
      func.setArgAttr(blockArg.getArgNumber(), "irSynth.lastStoredMemref",
                      UnitAttr::get(op.getContext()));
    }
  });
}

std::unique_ptr<OperationPass<ModuleOp>>
createAnnotateLastStoredMemrefArgPass() {
  return std::make_unique<AnnotateLastStoredMemrefArgPass>();
}
