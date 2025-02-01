#include "bindings/c/MlirSynthCApi.h"
#include "bindings/c/MlirSynthPasses.h"

#include "mlir-c/Support.h"
#include "mlir/Bindings/Python/PybindAdaptors.h"
#include "llvm/Support/raw_ostream.h"

#include "mlir-c/Bindings/Python/Interop.h"
#include "mlir-c/Pass.h"
#include "transforms/Passes.h"

#include <iostream>
#include <pybind11/pytypes.h>

namespace py = pybind11;
using namespace mlir;
using namespace mlir::python;
using namespace mlir::python::adaptors;

SynthesisOptions parseSynthesisOptions(py::dict &options);

PYBIND11_MODULE(_synth, m) {
  auto synth_m = m.def_submodule("synth");

  synth_m.def("register_passes", []() { registerMlirSynthPasses(); });

  synth_m.def(
      "register_dialects",
      [](MlirContext context) { registerDialects(context); },
      py::arg("context") = py::none());

  synth_m.def(
      "lower_chlo_to_affine",
      [](MlirModule &module, bool expandArithOps) {
        lowerCHLOToAffine(module, expandArithOps);
      },
      py::arg("module"), py::arg("expand_arith_ops"));

  synth_m.def(
      "emit_python",
      [](MlirModule &module) {
        std::string pyCode;
        emitPython(module, wrap(&pyCode));
        return py::str(pyCode);
      },
      py::arg("module"));

  synth_m.def(
      "predict_ops",
      [](MlirOperation &op) {
        std::string ops;
        predictGuideOps(op, wrap(&ops));
        return py::str(ops);
      },
      py::arg("module"));

  synth_m.def(
      "predict_constants",
      [](MlirOperation &op) {
        std::string constants;
        predictConstants(op, wrap(&constants));
        return py::str(constants);
      },
      py::arg("module"));

  synth_m.def(
      "enumerate_one_op",
      [](py::list &pyOperations, py::list &pyArgTypes, MlirType pyReturnType,
         py::dict &options, py::list &resultModulesStrs) {
        // Parse args
        // - Operation names
        auto opsVec = pyOperations.cast<std::vector<std::string>>();
        llvm::SmallVector<MlirStringRef> ops;
        for (auto &op : opsVec)
          ops.push_back(mlirStringRefCreate(op.c_str(), op.size()));

        // - ArgTypes
        auto argTypesVec = pyArgTypes.cast<std::vector<MlirType>>();
        llvm::SmallVector<MlirType> argTypes;
        for (auto &type : argTypesVec)
          argTypes.push_back(type);

        // - Return type
        MlirType returnType = pyReturnType;

        // Parse SynthesisOptions from Python dict
        SynthesisOptions opts = parseSynthesisOptions(options);

        llvm::SmallVector<std::string> resultModuleStrs;
        enumerateOneOp(ops.data(), ops.size(), argTypes.data(), argTypes.size(),
                       returnType, opts, (void *)&resultModuleStrs);

        // Convert result modules to Python strings
        for (auto &resultModuleStr : resultModuleStrs) {
          resultModulesStrs.append(py::str(resultModuleStr));
        }
      },
      py::arg("operations"), py::arg("argTypes"), py::arg("returnType"),
      py::arg("options"), py::arg("resultModulesStrs"));
}

SynthesisOptions parseSynthesisOptions(py::dict &options) {
  SynthesisOptions opts;
  if (options.contains("printStatusNames"))
    opts.printStatusTiles = options["printStatusNames"].cast<bool>();
  if (options.contains("printStatusTiles"))
    opts.printStatusTiles = options["printStatusTiles"].cast<bool>();
  if (options.contains("printValidCandidates"))
    opts.printValidCandidates = options["printValidCandidates"].cast<bool>();
  if (options.contains("printInvalidCandidates"))
    opts.printInvalidCandidates =
        options["printInvalidCandidates"].cast<bool>();
  if (options.contains("printStats"))
    opts.printStats = options["printStats"].cast<bool>();
  if (options.contains("printArgsAndResults"))
    opts.printArgsAndResults = options["printArgsAndResults"].cast<bool>();
  if (options.contains("printSynthesisSteps"))
    opts.printSynthesisSteps = options["printSynthesisSteps"].cast<bool>();
  if (options.contains("maxNumOps"))
    opts.maxNumOps = options["maxNumOps"].cast<int>();
  if (options.contains("timeoutPerFunction"))
    opts.timeoutPerFunction = options["timeoutPerFunction"].cast<int>();
  if (options.contains("ignoreEquivalentCandidates"))
    opts.ignoreEquivalentCandidates =
        options["ignoreEquivalentCandidates"].cast<bool>();
  if (options.contains("ignoreTypes"))
    opts.ignoreTypes = options["ignoreTypes"].cast<bool>();
  if (options.contains("skipTypeInference"))
    opts.skipTypeInference = options["skipTypeInference"].cast<bool>();
  if (options.contains("skipMergeCandidateArguments"))
    opts.skipMergeCandidateArguments =
        options["skipMergeCandidateArguments"].cast<bool>();
  if (options.contains("stopOnSolutionCandidate"))
    opts.stopOnSolutionCandidate =
        options["stopOnSolutionCandidate"].cast<bool>();
  if (options.contains("withCopyArgs"))
    opts.withCopyArgs = options["withCopyArgs"].cast<bool>();
  if (options.contains("guide"))
    opts.guide = options["guide"].cast<bool>();
  if (options.contains("distribute"))
    opts.distribute = options["distribute"].cast<bool>();
  return opts;
}