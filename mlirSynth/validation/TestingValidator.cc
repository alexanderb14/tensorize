#include "TestingValidator.h"

#include "execution/ArrayUtils.h"
#include "execution/Lowering.h"
#include "synthesis/Synthesizer.h"
#include "synthesis/Utils.h"
#include "transforms/Utils.h"

using namespace mlir;

bool testValidate(ModuleOp lhsModule, ModuleOp rhsModule,
                  bool printArgsAndResults, bool printResults) {
  // LHS
  auto lhsFunctions = getFunctions(lhsModule);
  assert(lhsFunctions.size() == 1 && "Expected exactly one function");
  auto lhsFunction = lhsFunctions.front();

  auto *ctx = lhsFunction->getContext();

  lhsFunction->setAttr("llvm.emit_c_interface", UnitAttr::get(ctx));
  lhsFunction.setSymName("foo");

  // Create inputs.
  auto args = createArgs(lhsFunction);
  randomlyInitializeArgs(lhsFunction, args);
  auto targetShape = getReturnType(lhsFunction).cast<ShapedType>().getShape();

  // Lower and run the lhs function on the inputs.
  auto pm = std::make_shared<mlir::PassManager>(ctx);
  Polygeist::addAffineToLLVMPasses(pm);
  assert(succeeded(pm->run(lhsModule)) &&
         "Failed to lower affine to LLVM dialect");

  auto refRet = getOwningMemRefForShape(targetShape);
  assert(succeeded(jitAndInvoke(lhsModule, args, refRet)));
  double *refOut = getReturnDataPtr(refRet);

  if (printArgsAndResults)
    printArgsAndResultsInPython(args, refOut, targetShape);

  // RHS
  auto rhsFunctions = getFunctions(rhsModule);
  assert(rhsFunctions.size() == 1 && "Expected exactly one function");
  auto rhsFunction = rhsFunctions.front();

  rhsFunction->setAttr("llvm.emit_c_interface", UnitAttr::get(ctx));
  rhsFunction.setSymName("foo");

  // Lower and run the rhs function on the inputs.
  auto pmRHS = std::make_shared<mlir::PassManager>(ctx);
  HLO::addAffineToLLVMPasses(pmRHS);
  assert(succeeded(pm->run(rhsModule)) &&
         "Failed to lower chlo to LLVM dialect");

  auto rhsRet = getOwningMemRefForShape(targetShape);
  convertScalarToMemrefArgs(args);
  assert(succeeded(jitAndInvoke(rhsModule, args, rhsRet)));
  double *rhsOut = getReturnDataPtr(rhsRet);

  if (printResults) {
    printArray(refOut, targetShape, llvm::outs());
    llvm::outs() << "\n";
    printArray(rhsOut, targetShape, llvm::outs());
    llvm::outs() << "\n";
  }

  // Test for equivalence.
  return areArraysEqual(refOut, rhsOut, targetShape);
}
