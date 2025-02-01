#include "transforms/Passes.h"

#include "transforms/Utils.h"

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

#define GEN_PASS_DEF_COPYMODIFIEDMEMREFS
#include "transforms/Passes.h.inc"

namespace {

struct CopyModifiedMemrefsPass
    : public impl::CopyModifiedMemrefsBase<CopyModifiedMemrefsPass> {
  void runOnOperation() override;
};

}  // namespace
}  // namespace mlir

void copyModifiedMemrefs(func::FuncOp &op) {
  // Only do it on functions with the "irsynth.original" attribute.
  if (!op->hasAttr("irsynth.original"))
    return;

  // Set insertion point to the start of the loop.
  auto funcs = getTopLevelLoops(op);
  assert(funcs.size() == 1 && "Expected only one top level loop");
  OpBuilder builder(op->getContext());
  builder.setInsertionPoint(funcs[0]);

  // Get all stored memref values.
  llvm::SetVector<Value> storedMemrefs;
  op->walk([&](AffineStoreOp op) { storedMemrefs.insert(op.getMemRef()); });

  // Copy all stored memref values and put them into the top of the function.
  for (auto value : storedMemrefs) {
    // Create a new memref with the name of the old one.
    auto memreftype = value.getType().cast<MemRefType>();
    auto newMemRef = builder.create<memref::AllocOp>(op.getLoc(), memreftype);

    // Replace all uses of the old memref with the new one after the copy except
    // for the newly created copy.
    value.replaceAllUsesExcept(newMemRef, newMemRef);

    // Copy the value.
    builder.create<memref::CopyOp>(op.getLoc(), value, newMemRef);
  }
}

void CopyModifiedMemrefsPass::runOnOperation() {
  auto operation = getOperation();
  for (auto func : operation.getOps<func::FuncOp>())
    copyModifiedMemrefs(func);
}

std::unique_ptr<OperationPass<ModuleOp>> createCopyModifiedMemrefsPass() {
  return std::make_unique<CopyModifiedMemrefsPass>();
}
