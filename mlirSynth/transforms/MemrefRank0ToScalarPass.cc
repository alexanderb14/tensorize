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

#define GEN_PASS_DEF_MEMREFRANK0TOSCALAR
#include "transforms/Passes.h.inc"

namespace {

struct MemrefRank0ToScalarPass
    : public impl::MemrefRank0ToScalarBase<MemrefRank0ToScalarPass> {
  void runOnOperation() override;
};

}  // namespace
}  // namespace mlir

void convertRank0MemrefsToScalars(func::FuncOp &func) {
  // Traverse functions arguments. If there's a rank 0 memref, replace it with a
  // scalar. Then look for all uses and replace them.
  auto funcArgs = func.getArguments();
  for (auto [argIdx, arg] : llvm::enumerate(funcArgs)) {
    auto argType = arg.getType();
    if (argType.isa<MemRefType>()) {
      auto memrefType = argType.cast<MemRefType>();
      if (memrefType.getShape().empty()) {
        // Replace memref type in arg with scalar.
        auto scalarType = memrefType.getElementType();
        arg.setType(scalarType);

        auto argTypes = func.getArgumentTypes().vec();
        argTypes[argIdx] = scalarType;

        func.setFunctionType(mlir::FunctionType::get(
            func->getContext(), argTypes, func.getResultTypes()));

        // Replace all uses of memref with scalar.
        auto argUsers = arg.getUsers();
        SmallVector<mlir::Operation *> argUsersCopy(argUsers.begin(),
                                                    argUsers.end());
        for (auto *argUserOp : argUsersCopy) {
          // If the user is a load, replace the loads uses with the scalar.
          if (auto loadOp = dyn_cast<memref::LoadOp>(argUserOp)) {
            auto loadUses = loadOp->getUses();
            for (auto &loadUse : loadUses) {
              auto *loadUser = loadUse.getOwner();
              loadUser->replaceUsesOfWith(loadOp, arg);
            }

            // Then delete the load.
            loadOp->erase();
          }
        }
      }
    }
  }
}

void MemrefRank0ToScalarPass::runOnOperation() {
  auto module = getOperation();

  module.walk([&](func::FuncOp func) { convertRank0MemrefsToScalars(func); });
}

std::unique_ptr<OperationPass<ModuleOp>> createMemrefRank0ToScalarPass() {
  return std::make_unique<MemrefRank0ToScalarPass>();
}