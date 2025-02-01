#include "transforms/Passes.h"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::bufferization;

namespace mlir {

#define GEN_PASS_DEF_FOLDTOTENSORTOMEMREFPAIR
#include "transforms/Passes.h.inc"

namespace {

struct FoldToTensorToMemrefPairPass
    : public impl::FoldToTensorToMemrefPairBase<FoldToTensorToMemrefPairPass> {
  void runOnOperation() override;
};

}  // namespace
}  // namespace mlir

struct FoldToTensorToMemrefPair : public RewritePattern {
  FoldToTensorToMemrefPair(MLIRContext *context)
      : RewritePattern(ToTensorOp::getOperationName(), 0, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    ToTensorOp toTensor = cast<ToTensorOp>(op);

    auto tensorToMemref = toTensor.getMemref().getDefiningOp<ToMemrefOp>();
    if (!tensorToMemref)
      return failure();

    // Rewrite if the type did not change.
    Type srcType = tensorToMemref.getTensor().getType();
    Type destType = toTensor.getType();

    if (srcType == destType) {
      rewriter.replaceOp(toTensor, tensorToMemref.getTensor());
      return success();
    }

    return failure();
  }
};

void FoldToTensorToMemrefPairPass::runOnOperation() {
  auto op = getOperation();

  RewritePatternSet patterns(op.getContext());
  patterns.add<FoldToTensorToMemrefPair>(op.getContext());

  (void)applyPatternsAndFoldGreedily(op, std::move(patterns));
}

std::unique_ptr<OperationPass<ModuleOp>> createFoldToTensorToMemrefPairPass() {
  return std::make_unique<FoldToTensorToMemrefPairPass>();
}
