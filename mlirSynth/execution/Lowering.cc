#include "Lowering.h"

#include "gml_st/transforms/passes.h"
#include "mlir-hlo/Dialect/mhlo/transforms/passes.h"
#include "mlir-hlo/Transforms/passes.h"
#include "mlir/Bytecode/BytecodeReader.h"
#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/AsyncToLLVM/AsyncToLLVM.h"
#include "mlir/Conversion/BufferizationToMemRef/BufferizationToMemRef.h"
#include "mlir/Conversion/ComplexToLLVM/ComplexToLLVM.h"
#include "mlir/Conversion/ComplexToStandard/ComplexToStandard.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/LinalgToLLVM/LinalgToLLVM.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/MathToLibm/MathToLibm.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/SPIRVToLLVM/SPIRVToLLVMPass.h"
#include "mlir/Conversion/ShapeToStandard/ShapeToStandard.h"
#include "mlir/Conversion/TensorToLinalg/TensorToLinalgPass.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "mlir/Conversion/VectorToSCF/VectorToSCF.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Async/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/Shape/Transforms/Passes.h"
#include "mlir/Dialect/Tensor/Transforms/Passes.h"
#include "mlir/ExecutionEngine/CRunnerUtils.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/MemRefUtils.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace HLO {
void addCHLOToAffinePasses(std::shared_ptr<mlir::PassManager> pm, bool expandArithOps) {
  pm->addPass(mhlo::createStablehloLegalizeToHloPass());
  pm->addNestedPass<func::FuncOp>(mhlo::createChloLegalizeToHloPass());

  // Note: The below MHLO -> LLVM lowering pipeline has been copied from:
  // TensorFlow@d8ae1c040f73ffdbf77e005b7d2909976fa4b31f
  // compiler/xla/service/cpu/cpu_compiler.cc
  // All commented calls are part of XLA and not available in MLIR-HLO.

  // proved statically and changed to const witness) early to allow more
  // efficient broadcast operations moving.
  // Move up broadcasting operations to allow for more fusion opportunities.
  pm->addPass(mlir::createInlinerPass());
  pm->addPass(mlir::mhlo::createExpandHloTuplesPass("main"));
  // TODO(b/233771980): Remove once custom_call doesn't use tuples.
  pm->addNestedPass<mlir::func::FuncOp>(mlir::mhlo::createFlattenTuplePass());
  //  pm->addPass(createXlaAbiLegalizationPass());
  pm->addNestedPass<mlir::func::FuncOp>(
      mlir::mhlo::createLegalizeGeneralDotPass());
  pm->addNestedPass<mlir::func::FuncOp>(
      mlir::mhlo::createBroadcastPropagationPass());
  pm->addPass(mlir::createCSEPass());
  pm->addPass(mlir::createCanonicalizerPass());

  // Transform HLO operations to Linalg.
  pm->addNestedPass<mlir::func::FuncOp>(mlir::mhlo::createLegalizeSortPass());
  pm->addNestedPass<mlir::func::FuncOp>(
      mlir::mhlo::createLegalizeControlFlowPass());
  pm->addPass(::mlir::mhlo::createLegalizeToArithmeticPass());
  pm->addNestedPass<mlir::func::FuncOp>(
      mlir::mhlo::createLegalizeHloToLinalgPass());
  pm->addNestedPass<mlir::func::FuncOp>(
      mlir::mhlo::createLegalizeMHLOToTHLOPass());

  // Lower index cast on tensors to tensor.generate.
  pm->addNestedPass<mlir::func::FuncOp>(mlir::createLowerIndexCastPass());

  pm->addPass(mlir::mhlo::createConvertToSignlessPass());

  // Transform scatter ops.
  pm->addNestedPass<mlir::func::FuncOp>(
      mlir::gml_st::createTransformScatterForCpuPass());

  // Lower shape dialect to standard to enable linalg canonicalizations (e.g.
  // use linalg inputs instead of outputs for memref.dim operations).
  pm->addNestedPass<mlir::func::FuncOp>(mlir::createShapeSimplification());
  pm->addNestedPass<mlir::func::FuncOp>(mlir::createShapeToShapeLowering());
  pm->addPass(mlir::createConvertShapeToStandardPass());
  pm->addNestedPass<mlir::func::FuncOp>(
      mlir::createConvertShapeConstraintsPass());

  // Fuse Linalg on tensors operations.
  pm->addPass(mlir::createCSEPass());
  pm->addPass(mlir::memref::createResolveShapedTypeResultDimsPass());
  pm->addPass(mlir::createCanonicalizerPass());
  pm->addNestedPass<mlir::func::FuncOp>(
      mlir::createLinalgElementwiseOpFusionPass());
  pm->addPass(mlir::createReconcileUnrealizedCastsPass());
  pm->addPass(mlir::createConvertTensorToLinalgPass());

  // Detensorize SCF iter args.
  pm->addNestedPass<mlir::func::FuncOp>(mlir::createDetensorizeScfOpsPass());
  // mhlo ops on unit tensors generate trivial linalg.generics, which
  // one-shot-bufferize generates unnecessary allocs for. The detensorize pass
  // replaces these linalg.generics with scalar ops.
  auto detensorize = mlir::createLinalgDetensorizePass();
  if (detensorize->initializeOptions("aggressive-mode=true").failed()) {
    llvm::outs() << "Failed to set up detensorize pass.";
  }
  pm->addNestedPass<mlir::func::FuncOp>(std::move(detensorize));
  pm->addNestedPass<mlir::func::FuncOp>(mlir::createScalarizationPass());
  pm->addNestedPass<mlir::func::FuncOp>(
      mlir::bufferization::createEmptyTensorToAllocTensorPass());

  // Always run canonicalizer (which does dead code removal) before
  // bufferizing anything.
  pm->addPass(mlir::createCanonicalizerPass());

  pm->addPass(mlir::hlo::createOneShotBufferizePass());

  // Handle framework specific requirements for buffers and then insert
  // deallocations for temporary buffers.
  pm->addNestedPass<mlir::func::FuncOp>(mlir::createConvertLinalgToLoopsPass());
  pm->addNestedPass<mlir::func::FuncOp>(mlir::gml_st::createGmlStToScfPass());
  pm->addPass(mlir::createCSEPass());
  pm->addPass(mlir::createCanonicalizerPass());
  //  pm->addPass(mlir::bufferization::createBufferResultsToOutParamsPass());
  //  if (options.outline_with_xla_framework) {
  //    pm->addPass(mlir::mhlo::CreateOutlineWithXLAFrameworkPass());
  //  }
  pm->addPass(mlir::createInlinerPass());
  pm->addNestedPass<mlir::func::FuncOp>(
      mlir::bufferization::createBufferDeallocationPass());

  pm->addPass(mlir::createBufferizationToMemRefPass());

  // Specialize linalg.matmul to linalg.dot, linalg.matvec or linalg.vecmat,
  // and immediately canonicalize to clean up not taken branches.
  // pm->addNestedPass<mlir::func::FuncOp>(CreateLinalgMatmulSpecializationPass());
  pm->addPass(mlir::createCanonicalizerPass());

  // Tile and vectorize linalg operation using Linalg Codegen Strategy.
  pm->addNestedPass<mlir::func::FuncOp>(
      mlir::createConvertComplexToStandardPass());

  pm->addPass(mlir::createCSEPass());
  pm->addPass(mlir::createCanonicalizerPass());

  mlir::VectorTransferToSCFOptions vec_to_scf_options;
  vec_to_scf_options.unroll = true;
  pm->addNestedPass<mlir::func::FuncOp>(
      mlir::createConvertVectorToSCFPass(vec_to_scf_options));

  if (expandArithOps) {
    pm->addNestedPass<mlir::func::FuncOp>(
        mlir::arith::createArithExpandOpsPass());
  }
  pm->addNestedPass<mlir::func::FuncOp>(mlir::memref::createExpandOpsPass());
}

void addAffineToLLVMPasses(std::shared_ptr<mlir::PassManager> pm) {
  //  pm->addPass(mlir::mhlo::CreateLegalizeXLAFrameworkToLLVMPass());
  pm->addPass(createLowerAffinePass());
  pm->addPass(mlir::hlo::createGenericHostToLLVMPass());
  pm->addPass(mlir::createReconcileUnrealizedCastsPass());
}

} // namespace HLO

namespace Polygeist {
void addAffineToLLVMPasses(std::shared_ptr<mlir::PassManager> pm) {
  pm->addPass(createLowerAffinePass());
  pm->addPass(createConvertSCFToCFPass());
  pm->addPass(cf::createConvertControlFlowToLLVMPass());
  pm->addPass(mlir::createMemRefToLLVMConversionPass());
  pm->addNestedPass<func::FuncOp>(mlir::createArithToLLVMConversionPass());
  pm->addPass(createConvertMathToLLVMPass());
  pm->addPass(createConvertFuncToLLVMPass());
  pm->addPass(mlir::createReconcileUnrealizedCastsPass());
}
} // namespace Polygeist
