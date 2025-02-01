#include "transforms/Passes.h"

#include "analysis/PolyhedralAnalysis.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "transforms/Utils.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir/Pass/AnalysisManager.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/ADT/SetVector.h"

using namespace mlir;

namespace mlir {

#define GEN_PASS_DEF_LOOPOUTLINE
#include "transforms/Passes.h.inc"

namespace {

struct LoopOutlinePass : public impl::LoopOutlineBase<LoopOutlinePass> {
  void runOnOperation() override;
};

} // namespace
} // namespace mlir

BlockAndValueMapping reverseMap(BlockAndValueMapping &mapper) {
  BlockAndValueMapping reverseMapper;
  for (auto &pair : mapper.getValueMap())
    reverseMapper.map(pair.second, pair.first);
  return reverseMapper;
}

void outlineLoops(func::FuncOp &origFunc) {
  auto unknownLoc = UnknownLoc::get(origFunc.getContext());

  bool debug = false;
  if (debug)
    origFunc.dump();

  auto module = origFunc->getParentOfType<ModuleOp>();
  auto topLoops = getTopLevelLoops(origFunc);
  auto builder = OpBuilder::atBlockBegin(module.getBody());

  BlockAndValueMapping fnResultMapper;
  Operation *lastFunc = nullptr;

  unsigned loopCounter = 0;
  for (auto *topLoop : topLoops) {
    auto loop = cast<AffineForOp>(topLoop);

    auto undefinedValues = getOutOfBlockDefValues(loop.getBody());
    auto loadedValues = getLoadedMemRefValues(topLoop);
    auto storedValues = getStoredMemRefValues(topLoop);
    auto allocaValues = getAllocaMemRefValues(topLoop);

    if (debug) {
      llvm::outs() << "-----------------\n";
      topLoop->dump();
      llvm::outs() << "Undefined values:\n";
      for (auto value : undefinedValues)
        value.dump();
      llvm::outs() << "Loaded:\n";
      for (auto value : loadedValues)
        value.dump();
      llvm::outs() << "Stored:\n";
      for (auto value : storedValues)
        value.dump();
      llvm::outs() << "Alloca:\n";
      for (auto value : allocaValues)
        value.dump();
    }

    // Substract alloca values from loaded and stored values.
    // We don't want to add alloca values as arguments to the new function.
    for (auto value : allocaValues) {
      loadedValues.remove(value);
      storedValues.remove(value);
    }

    // Create a new function.
    // ---------------------------------------------
    OpBuilder builder(origFunc.getContext());
    auto func = builder.create<func::FuncOp>(
        unknownLoc, "fn_" + std::to_string(loopCounter++),
        builder.getFunctionType({}, {}));
    func->setAttr("irsynth.original", builder.getUnitAttr());
    auto &bodyBlock = *func.addEntryBlock();

    // Add arguments to function.
    BlockAndValueMapping argMapper;

    // - Add loaded values as arguments.
    for (auto value : loadedValues) {
      // Check if value is in the undefined values vector.
      if (std::find(undefinedValues.begin(), undefinedValues.end(), value) !=
          undefinedValues.end())
        continue;
      auto newArg = bodyBlock.addArgument(value.getType(), unknownLoc);
      argMapper.map(value, newArg);
    }
    // - Add undefined values as arguments or as local variables if they are
    // constants.
    for (auto value : undefinedValues) {
      if (argMapper.contains(value)) {
        continue;
      }

      // If the defining operation is a constant or a memref alloca, copy and
      // add it to the new function. Else, add it as an argument.
      auto *definingOp = value.getDefiningOp();
      // - Constant.
      if (definingOp && dyn_cast<arith::ConstantOp>(definingOp)) {
        auto constantOp = dyn_cast<arith::ConstantOp>(definingOp);
        auto newConstantOp = constantOp.clone();
        bodyBlock.push_back(newConstantOp);
        argMapper.map(value, newConstantOp.getResult());

        // - Memref alloca.
      } else if (definingOp && dyn_cast<memref::AllocaOp>(definingOp)) {
        auto allocaOp = dyn_cast<memref::AllocaOp>(definingOp);
        auto newAllocaOp = allocaOp.clone();
        bodyBlock.push_back(newAllocaOp);
        argMapper.map(value, newAllocaOp.getResult());

        // - Check if the memref alloca is followed by a constant and a store.
        // If so, add them to the new function.
        auto *nextOp = allocaOp->getNextNode();
        if (nextOp && dyn_cast<arith::ConstantOp>(nextOp)) {
          auto *nextNextOp = nextOp->getNextNode();
          if (nextNextOp && dyn_cast<AffineStoreOp>(nextNextOp)) {
            auto *newConstantOp = nextOp->clone();
            bodyBlock.push_back(newConstantOp);
            argMapper.map(nextOp->getResult(0), newConstantOp->getResult(0));

            auto *newStoreOp = nextNextOp->clone(argMapper);
            bodyBlock.push_back(newStoreOp);
          }
        }

        // Else, add as argument.
      } else {
        auto newArg = bodyBlock.addArgument(value.getType(), unknownLoc);
        argMapper.map(value, newArg);
      }
    }
    auto reverseMapper = reverseMap(argMapper);

    // Add body.
    bodyBlock.push_back(topLoop->clone(argMapper));

    // Add the last stored memref value as result.
    llvm::SetVector<Value> storedValue;

    Value lastStoredMemref = nullptr;
    topLoop->walk(
        [&](AffineStoreOp storeOp) { lastStoredMemref = storeOp.getMemref(); });
    assert(lastStoredMemref != nullptr && "No last stored memref found.");

    storedValue.insert(lastStoredMemref);

    // - Create return operation.
    llvm::SmallVector<Value> results;
    for (auto value : storedValue)
      results.push_back(argMapper.lookup(value));
    builder.setInsertionPoint(&bodyBlock, bodyBlock.end());
    builder.create<func::ReturnOp>(unknownLoc, results);

    // - Add the results to function type.
    llvm::SmallVector<Type> resultTypes;
    for (auto value : storedValue)
      resultTypes.push_back(value.getType());
    func.setFunctionType(
        builder.getFunctionType(bodyBlock.getArgumentTypes(), resultTypes));

    // - Add arg attributes of the original function.
    llvm::SmallVector<Attribute> argAttrs;
    if (auto origFuncArgAttrs = origFunc.getAllArgAttrs()) {
      for (auto arg : bodyBlock.getArguments()) {
        auto newArg = reverseMapper.lookup(arg);
        if (auto value = newArg.dyn_cast<BlockArgument>()) {
          auto attribute = origFuncArgAttrs[value.getArgNumber()];
          argAttrs.push_back(attribute);
        } else {
          argAttrs.push_back(builder.getDictionaryAttr({}));
        }
      }
      func.setAllArgAttrs(argAttrs);
    }

    // Insert the new function and replace the loop with a call to it.
    // ---------------------------------------------
    if (!lastFunc)
      builder.setInsertionPointToStart(module.getBody());
    else
      builder.setInsertionPointAfter(lastFunc);

    builder.insert(func);
    lastFunc = func;

    // Create args for the call.
    llvm::SmallVector<Value> args;
    for (auto arg : func.getArguments()) {
      auto value = reverseMapper.lookupOrNull(arg);

      // If the arg value was already recomputed by an earlier call, use this
      // one.
      if (fnResultMapper.contains(value))
        args.push_back(fnResultMapper.lookup(value));
      else
        args.push_back(value);
    }

    // Create function call.
    builder.setInsertionPoint(topLoop);
    auto callOp = builder.create<func::CallOp>(unknownLoc, func.getSymName(),
                                               func.getResultTypes(), args);

    // Add function call results to the fnResultMapper.
    for (unsigned i = 0; i < callOp.getNumResults(); i++)
      fnResultMapper.map(storedValue[i], callOp.getResult(i));

    // Remove the loop.
    topLoop->erase();
  }
}

void LoopOutlinePass::runOnOperation() {
  auto operation = getOperation();
  for (auto func : operation.getOps<func::FuncOp>())
    outlineLoops(func);
}

std::unique_ptr<OperationPass<ModuleOp>> createLoopOutlinePass() {
  return std::make_unique<LoopOutlinePass>();
}
