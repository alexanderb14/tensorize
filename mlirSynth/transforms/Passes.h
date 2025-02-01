#ifndef IRSYNTH_PASSES_H
#define IRSYNTH_PASSES_H

#include <functional>
#include <memory>

#include "mlir/Pass/Pass.h"

namespace mlir {
class ModuleOp;
class MLIRContext;
class ConversionTarget;
class DialectRegistry;
class PassManager;

namespace func {
class FuncOp;
}  // namespace func
} // namespace mlir

#define GEN_PASS_DECL
#include "transforms/Passes.h.inc"

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createAnnotateLastStoredMemrefArgPass();

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createChangeSizesPass();

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createCleanupPass();

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createCopyModifiedMemrefsPass();

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createFoldToTensorToMemrefPairPass();

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createLoopDistributionPass();

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createLoopOutlinePass();

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createMemrefToScfPass();

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createMemrefRank0ToScalarPass();

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createPrepareTargetPass();

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createTargetOutlinePass();

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

//#define GEN_PASS_DECL_TILELOOPSPASS
//#include "mlir-hlo/Transforms/passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "transforms/Passes.h.inc"

#endif  // IRSYNTH_PASSES_H
