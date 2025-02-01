#include "Candidate.h"

using namespace llvm;
using namespace mlir;

SmallVector<Value> Candidate::merge(MLIRContext &ctx,
                                    std::vector<CandidatePtr> &others,
                                    bool skipMergeCandidateArguments) {
  // Merge other candidates into this one.
  BlockAndValueMapping mapping;
  mlir::DenseMap<unsigned, BlockArgument> seenArgs;

  SmallVector<Value> resultValues = {};

  for (auto &other : others) {
    auto &block = region->getBlocks().front();
    OpBuilder builder(&ctx);
    builder.setInsertionPoint(&block, block.end());

    // Clone other candidates arguments
    SmallVector<Value> newArgs;
    for (auto arg :other->region->getArguments()) {

      auto newArg = block.addArgument(arg.getType(), UnknownLoc::get(&ctx));
      newArgs.push_back(newArg);
      mapping.map(arg, newArg);
    }

    // Clone other candidates operations, and update the operands with the
    // mapping. Also, add the result of the last operation to the resultValues.
    bool resultValueInserted = false;
    for (auto [opIdx, op] : llvm::enumerate(*other->getBlock())) {
      auto *clonedOp = builder.clone(op, mapping);
      mapping.map(op.getResults(), clonedOp->getResults());

      // Add the result to the resultValues if it is the last operation.
      if (opIdx == other->getBlock()->getOperations().size() - 1) {
        for (auto res : clonedOp->getResults()) {
          resultValues.push_back(res);
          resultValueInserted = true;
        }
      }
    }

    // If no result was inserted, add the just cloned argument as result. This
    // is the case when the candidate has no operations and only arguments.
    if (!resultValueInserted) {
      // Assert, because there should be only no-op candidates with one argument.
      assert(newArgs.size() == 1 && "Expected only one argument");

      for (auto res : newArgs) {
        resultValues.push_back(res);
      }
    }
  }

  assert (resultValues.size() == others.size()
          && "Result values size should be equal to the number of candidates");

  // Add other candidates op counters to own.
  for (auto &other : others) {
    numOps += other->getNumOps();
  }

  return resultValues;
}

void Candidate::addArgument(MLIRContext &ctx, Type type, unsigned argId) {
  argIds.push_back(argId);

  auto &block = region->getBlocks().front();
  unsigned argIdx = block.getNumArguments();
  block.insertArgument(argIdx, type, UnknownLoc::get(&ctx));
}

void Candidate::addOperation(MLIRContext &ctx, Operation *op, bool count) {
  auto &block = region->getBlocks().front();
  OpBuilder builder(&ctx);
  builder.setInsertionPoint(&block, block.end());
  builder.insert(op);

  if (count)
    numOps++;
}

SmallVector<Value> Candidate::getResults() {
  if (region->empty() || region->front().empty()) {
    // Return arguments.
    auto args = region->getBlocks().front().getArguments();
    return SmallVector<Value>(args.begin(), args.end());
  }
  // Return result of last op.
  return region->getBlocks().front().back().getResults();
}

void Candidate::dump() {
  llvm::outs() << "Candidate: \n";
  llvm::outs() << "- Args\n";
  unsigned argIdx = 0;
  for (auto &argId : argIds) {
    llvm::outs() << "  argId: " << argId << ", "
                 << getRegion()->getBlocks().front().getArgument(argIdx)
                 << "\n";
  }

  llvm::outs() << "- Ops\n";
  for (auto &op : region->getBlocks().front()) {
    llvm::outs() << "  ";
    op.dump();
  }
}
