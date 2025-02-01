#include "transforms/Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace mlir {

#define GEN_PASS_DEF_MEMREFTOSCF
#include "transforms/Passes.h.inc"

namespace {

struct MemrefToScfPass
    : public impl::MemrefToScfBase<MemrefToScfPass> {
  void runOnOperation() override;
};

}  // namespace
}  // namespace mlir

struct MemrefCopyToScf : public RewritePattern {
  MemrefCopyToScf(MLIRContext *ctx)
      : RewritePattern(memref::CopyOp::getOperationName(), 0, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto *ctx = op->getContext();
    auto copyOp = cast<memref::CopyOp>(op);

    // Get the source memref shape.
    auto sourceMemref = copyOp.getSource();
    auto sourceMemrefType = sourceMemref.getType().cast<MemRefType>();
    auto sourceShape = sourceMemrefType.getShape();

    // Create a loop for each dimension and create a memref.load from the source memref and a memref.store to the destination memref.
    auto builder = OpBuilder(copyOp);

    SmallVector<mlir::Value> indices;
    for (auto dimSize : sourceShape) {
      // Create for op.
      auto forOp = builder.create<scf::ForOp>(
          UnknownLoc::get(ctx),
          builder.create<arith::ConstantIndexOp>(UnknownLoc::get(ctx), 0),
          builder.create<arith::ConstantIndexOp>(UnknownLoc::get(ctx), dimSize),
          builder.create<arith::ConstantIndexOp>(UnknownLoc::get(ctx), 1));

      indices.push_back(forOp.getInductionVar());

      // Set insertion point inside body for next loop to be inserted.
      builder.setInsertionPointToStart(forOp.getBody());

      // If we are in the innermost loop, create the declaration.
      if (indices.size() == sourceShape.size()) {
        // Create load.
        mlir::Type lhsType =
            sourceMemrefType.getElementType();
        auto lhsLoad = builder.create<memref::LoadOp>(
            UnknownLoc::get(ctx), lhsType, sourceMemref, indices);

        // Create store.
        builder.create<memref::StoreOp>(UnknownLoc::get(ctx),
                                        lhsLoad, copyOp.getTarget(), indices);
      }
    }

    // Erase the original op.
    copyOp.erase();

    return success();
  }
};

struct MemrefSubviewToScf : public RewritePattern {
  MemrefSubviewToScf(MLIRContext *ctx)
      : RewritePattern(memref::SubViewOp::getOperationName(), 0, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto *ctx = op->getContext();
    auto subviewOp = cast<memref::SubViewOp>(op);

    auto sourceMemref = subviewOp.getSource();
    auto sourceMemrefType = sourceMemref.getType().cast<MemRefType>();
    auto sourceShape = sourceMemrefType.getShape();

    auto destMemref = subviewOp.getResult();
    auto destMemrefType = destMemref.getType().cast<MemRefType>();
    auto destShape = destMemrefType.getShape();

    auto builder = OpBuilder(subviewOp);

    auto offsets = subviewOp.getStaticOffsets();
    for (auto offsetValue : offsets) {
      int offset = offsetValue.cast<IntegerAttr>().getInt();
      if (offset != 0) {
        return failure();
      }
    }

    // Create a new memref with the destination shape.
    MemRefType newMemrefType = MemRefType::get(destShape, destMemrefType.getElementType());
    auto newMemref = builder.create<memref::AllocOp>(
        UnknownLoc::get(ctx), newMemrefType);

    // Create for loops for each dimension of the destination shape.
    SmallVector<mlir::Value> destIndices;
    for (auto dimSize : destShape) {
      // Create for op.
      auto forOp = builder.create<scf::ForOp>(
          UnknownLoc::get(ctx),
          builder.create<arith::ConstantIndexOp>(UnknownLoc::get(ctx), 0),
          builder.create<arith::ConstantIndexOp>(UnknownLoc::get(ctx), dimSize),
          builder.create<arith::ConstantIndexOp>(UnknownLoc::get(ctx), 1));

      destIndices.push_back(forOp.getInductionVar());

      // Set insertion point inside body for next loop to be inserted.
      builder.setInsertionPointToStart(forOp.getBody());
    }

    // - Load and store.
    auto lhsLoad = builder.create<memref::LoadOp>(
        UnknownLoc::get(ctx), sourceMemrefType.getElementType(), sourceMemref, destIndices);

    builder.create<memref::StoreOp>(UnknownLoc::get(ctx),
                                    lhsLoad, newMemref.getResult(), destIndices);

    // Replace all uses of the original op with the new memref.
    subviewOp->replaceAllUsesWith(newMemref);

    // Remove the original op.
    subviewOp.erase();

    return success();
  }
};

template <typename OpTy>
void toLoops(Operation* op, bool invert) {
  auto *ctx = op->getContext();
  auto expandShapeOp = cast<OpTy>(op);

  // Get the source and destination memref shapes.
  auto sourceMemref = expandShapeOp.getSrc();
  auto sourceMemrefType = sourceMemref.getType().template cast<MemRefType>();
  auto sourceShape = sourceMemrefType.getShape();

  auto destMemref = expandShapeOp.getResultType();
  auto destMemrefType = destMemref.template cast<MemRefType>();
  auto destShape = destMemrefType.getShape();

  if (invert) {
    std::swap(sourceShape, destShape);
  }

  auto builder = OpBuilder(expandShapeOp);

  // Create a new memref with the destination shape.
  auto newMemref = builder.create<memref::AllocOp>(
      UnknownLoc::get(ctx), destMemrefType);

  // Create for loops for each dimension of the destination shape.
  SmallVector<mlir::Value> destIndices;
  for (auto dimSize : destShape) {
    // Create for op.
    auto forOp = builder.create<scf::ForOp>(
        UnknownLoc::get(ctx),
        builder.create<arith::ConstantIndexOp>(UnknownLoc::get(ctx), 0),
        builder.create<arith::ConstantIndexOp>(UnknownLoc::get(ctx), dimSize),
        builder.create<arith::ConstantIndexOp>(UnknownLoc::get(ctx), 1));

    destIndices.push_back(forOp.getInductionVar());

    // Set insertion point inside body for next loop to be inserted.
    builder.setInsertionPointToStart(forOp.getBody());
  }

  // Create statement.
  // - Calculate indices according to reassociation array.
  SmallVector<mlir::Value> sourceIndices;
  auto reassociation = expandShapeOp.getReassociation();
  for (auto reassIndicesAttr : reassociation) {
    mlir::Value lastIdx = nullptr;

    auto reassIndices = reassIndicesAttr.template cast<ArrayAttr>().getValue();
    for (auto idxAttr : reassIndices) {
      int idx = idxAttr.template cast<IntegerAttr>().getInt();

      if (lastIdx == nullptr) {
        lastIdx = destIndices[idx];
      } else {
        auto constOp = builder.create<arith::ConstantIndexOp>(
            UnknownLoc::get(ctx), destShape[idx]);
        auto mulOp = builder.create<arith::MulIOp>(
            UnknownLoc::get(ctx), constOp, lastIdx);

        auto addOp = builder.create<arith::AddIOp>(
            UnknownLoc::get(ctx), mulOp, destIndices[idx]);
        lastIdx = addOp.getResult();
      }

    }

    sourceIndices.push_back(lastIdx);
  }

  if (invert) {
    std::swap(sourceIndices, destIndices);
  }

  // - Load and store.
  auto lhsLoad = builder.create<memref::LoadOp>(
      UnknownLoc::get(ctx), sourceMemrefType.getElementType(), sourceMemref, sourceIndices);

  builder.create<memref::StoreOp>(UnknownLoc::get(ctx),
                                  lhsLoad, newMemref.getResult(), destIndices);

  // Replace all uses of the original op with the new memref.
  expandShapeOp->replaceAllUsesWith(newMemref);

  // Remove the original op.
  expandShapeOp.erase();
}

struct MemrefExpandShapeToScf : public RewritePattern {
  MemrefExpandShapeToScf(MLIRContext *ctx)
      : RewritePattern(memref::ExpandShapeOp::getOperationName(), 0, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    toLoops<memref::ExpandShapeOp>(op, false);
    return success();
  }
};

struct MemrefCollapseShapeToScf : public RewritePattern {
  MemrefCollapseShapeToScf(MLIRContext *ctx)
      : RewritePattern(memref::CollapseShapeOp::getOperationName(), 0, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    toLoops<memref::CollapseShapeOp>(op, true);
    return success();
  }
};

void MemrefToScfPass::runOnOperation() {
  auto op = getOperation();

  RewritePatternSet patterns(op.getContext());
  patterns.add<MemrefCopyToScf>(op.getContext());
  patterns.add<MemrefSubviewToScf>(op.getContext());
  patterns.add<MemrefExpandShapeToScf>(op.getContext());
  patterns.add<MemrefCollapseShapeToScf>(op.getContext());

  (void)applyPatternsAndFoldGreedily(op, std::move(patterns));
}

std::unique_ptr<OperationPass<ModuleOp>> createMemrefToScfPass() {
  return std::make_unique<MemrefToScfPass>();
}
