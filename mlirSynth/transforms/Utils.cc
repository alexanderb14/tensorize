#include "Utils.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

using namespace mlir;

std::vector<func::FuncOp> getFunctions(mlir::Operation *op,
                                       std::string attrName) {
  std::vector<func::FuncOp> functions;
  op->walk([&](func::FuncOp func) {
    if (attrName.empty() || func->getAttr(attrName))
      functions.push_back(func);
  });
  return functions;
}

llvm::SmallVector<mlir::Operation *> getTopLevelLoops(func::FuncOp &op) {
  llvm::SmallVector<mlir::Operation *> loops;
  assert(op.getBody().getBlocks().size() == 1);
  auto &block = op.getBody().getBlocks().front();
  for (auto &op : block.getOperations()) {
    if (dyn_cast<AffineForOp>(op)) {
      loops.push_back(&op);
    }
  }
  return loops;
}

llvm::SmallVector<Value> getOutOfBlockDefValues(mlir::Block *block) {
  // Get all values.
  llvm::DenseMap<Value, bool> allValues;
  // - Defined in the block.
  block->walk([&](Operation *op) {
    for (auto result : op->getResults())
      allValues[result] = true;
  });
  // - Defined as arguments.
  for (auto arg : block->getArguments())
    allValues[arg] = true;

  // Get all ops.
  llvm::SetVector<Operation *> allOps;
  block->walk([&](Operation *op) { allOps.insert(op); });

  llvm::SetVector<Value> undefinedValues;
  for (auto &op : allOps) {
    for (auto operand : op->getOperands()) {
      if (allValues.count(operand) == 0 && !operand.getType().isa<IndexType>())
        undefinedValues.insert(operand);
    }
  }

  // Convert to vector so that it has a stable order.
  llvm::SmallVector<Value> undefinedValuesVec;
  for (auto value : undefinedValues)
    undefinedValuesVec.push_back(value);

  return undefinedValuesVec;
}

llvm::SetVector<Value> getLoadedMemRefValues(mlir::Operation *op) {
  llvm::SetVector<Value> values;
  op->walk([&](AffineLoadOp loadOp) { values.insert(loadOp.getMemRef()); });
  return values;
}

llvm::SetVector<Value> getStoredMemRefValues(mlir::Operation *op) {
  llvm::SetVector<Value> values;
  op->walk([&](AffineStoreOp storeOp) { values.insert(storeOp.getMemRef()); });
  return values;
}

llvm::SetVector<mlir::Value> getAllocaMemRefValues(mlir::Operation *op) {
  llvm::SetVector<Value> values;
  op->walk(
      [&](memref::AllocaOp allocaOp) { values.insert(allocaOp.getResult()); });
  return values;
}
