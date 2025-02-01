#include "Synthesizer.h"

#include "common/Common.h"
#include "execution/ArgUtils.h"
#include "execution/ArrayUtils.h"
#include "mlir/IR/MLIRContext.h"
#include "synthesis/Candidate.h"
#include "synthesis/CandidateStore.h"
#include "synthesis/CartesianProduct.h"
#include "synthesis/Generators.h"
#include "synthesis/Grammar.h"
#include "synthesis/Options.h"
#include "synthesis/ProcessingStatus.h"
#include "synthesis/Spec.h"
#include "synthesis/Stats.h"
#include "synthesis/Threading.h"
#include "synthesis/Utils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Threading.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/LogicalResult.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "llvm/ADT/ArrayRef.h"

#include <chrono>
#include <cstdint>
#include <math.h>
#include <memory>
#include <mutex>
#include <optional>
#include <variant>

using namespace llvm;
using namespace mlir;

using operandsSetTy = std::vector<std::vector<CandidatePtr>>;
using attributesSetTy = std::vector<std::vector<mlir::Attribute>>;
using regionsSetTy = std::vector<std::vector<RegionPtr>>;

unsigned maxShapeRank = 4;

void printCandidate(ProcessingStatus status,
                    CandidateStorePtr &localCandidateStore,
                    CandidateStorePtr &candidateStore,
                    SynthesisOptions &options, SynthesisResultPtr &result,
                    std::mutex &printMutex) {
  // If there is nothing to print, return early.
  if (!(options.printStatusNames || options.printStatusTiles ||
        options.printValidCandidates || options.printInvalidCandidates)) {
    return;
  }

  // Build and print the status string.
  int candidateId;
  if (result) {
    candidateId = localCandidateStore->getCandidateId(result->candidate);
  } else {
    candidateId = -1;
  }

  std::string statusStr;
  bool printStatus = options.printStatusNames || options.printStatusTiles ||
                     options.printValidCandidates ||
                     options.printInvalidCandidates;
  if (printStatus) {
    if (options.printStatusTiles) {
      statusStr = " ";
    } else {
      statusStr = "Candidate " + std::to_string(candidateId) + ": ";

      statusStr += "status:" + processingStatusToStr(status);

      statusStr += ", preds:";
      bool first = true;
      for (auto &pred : result->candidate->getPredecessors()) {
        if (!first)
          statusStr += ",";
        first = false;
        statusStr += std::to_string(candidateStore->getCandidateId(pred));
      }
    }

    if (status == accept_as_candidate) {
      statusStr = "\033[1;42m" + statusStr + "\033[0m";
    } else {
      statusStr = "\033[1;41m" + statusStr + "\033[0m";
    }
  }

  // Print the module.
  if ((status == accept_as_candidate && options.printValidCandidates) ||
      (!(status == accept_as_candidate) && options.printInvalidCandidates) ||
      options.printStatusNames) {
    std::lock_guard<std::mutex> lock(printMutex);

    llvm::outs() << statusStr << "\n";
    if (result->module && status > reject_hasUnsupportedShapeRank) {
      result->module->print(llvm::outs());
    }
  }
}

void prepareInputFunction(func::FuncOp &inputFunction) {
  inputFunction->setAttr("llvm.emit_c_interface",
                         UnitAttr::get(inputFunction->getContext()));
  inputFunction.setName("foo");
}

void finalizeFunction(func::FuncOp func, std::string funcName) {
  func.setName(funcName);
  func->removeAttr("llvm.emit_c_interface");
  func->setAttr("irsynth.raised", UnitAttr::get(func->getContext()));
}

OwningOpRef<func::FuncOp> unwrapModule(ModuleOp &module) {
  std::vector<func::FuncOp> functions;
  module->walk([&](func::FuncOp func) { functions.push_back(func); });
  assert(functions.size() == 1);
  return functions[0];
}

OwningOpRef<ModuleOp> createModule(MLIRContext &ctx, func::FuncOp *function) {
  // Create an empty module.
  auto unknownLoc = UnknownLoc::get(&ctx);
  OwningOpRef<ModuleOp> module(ModuleOp::create(unknownLoc));

  // Create the builder, and set its insertion point in the module.
  OpBuilder builder(&ctx);
  auto &moduleBlock = module->getRegion().getBlocks().front();

  // Add the function to the module block.
  moduleBlock.push_back(function->getOperation()->clone());

  return module;
}

OwningOpRef<ModuleOp> createModule(MLIRContext &ctx, CandidatePtr candidate) {
  Region *region = candidate->getRegion();

  // Create an empty module.
  auto unknownLoc = UnknownLoc::get(&ctx);
  OwningOpRef<ModuleOp> module(ModuleOp::create(unknownLoc));

  // Create the builder, and set its insertion point in the module.
  OpBuilder builder(&ctx);
  auto &moduleBlock = module->getRegion().getBlocks().front();
  builder.setInsertionPoint(&moduleBlock, moduleBlock.begin());

  // Create function.
  auto func = builder.create<func::FuncOp>(
      unknownLoc, "foo", mlir::FunctionType::get(&ctx, {}, {}));
  func->setAttr("llvm.emit_c_interface", UnitAttr::get(&ctx));

  // Add the given region to the function.
  BlockAndValueMapping mapper;
  region->cloneInto(&func.getFunctionBody(), mapper);

  auto *bodyBlock = &func.getFunctionBody().getBlocks().front();
  builder.setInsertionPoint(bodyBlock, bodyBlock->end());

  // Add return to the function.
  if (bodyBlock->empty()) {
    // Is argument.
    builder.create<func::ReturnOp>(unknownLoc, bodyBlock->getArguments());
    func.setFunctionType(mlir::FunctionType::get(
        &ctx, bodyBlock->getArgumentTypes(), bodyBlock->getArgumentTypes()));
  } else {
    // Is operations.
    auto &lastOp = bodyBlock->back();
    builder.create<func::ReturnOp>(unknownLoc, lastOp.getResults());
    func.setFunctionType(mlir::FunctionType::get(
        &ctx, bodyBlock->getArgumentTypes(), lastOp.getResultTypes()));
  }

  //// Add argument indices to function.
  //for (unsigned argIdx = 0; argIdx < func.getNumArguments(); argIdx++) {
  //  func.setArgAttr(
  //      argIdx, "irsynth.fnArgIdx",
  //      IntegerAttr::get(IndexType::get(&ctx), candidate->getArgIds()[argIdx]));
  //}

  return copyModuleToCtx(&ctx, module.get());
}

LogicalResult verifyOp(Operation *op, RegisteredOperationName &opName) {
  if (failed(opName.verifyTraits(op))) {
    return failure();
  }
  return success();
}

LogicalResult inferResultTypes(MLIRContext &ctx, Operation *op) {
  // Infer the type.
  SmallVector<mlir::Type> inferredTypes;
  if (auto inferResultTypes = dyn_cast<InferTypeOpInterface>(op)) {
    if (failed(inferResultTypes.inferReturnTypes(
            &ctx, op->getLoc(), op->getOperands(), op->getAttrDictionary(),
            op->getRegions(), inferredTypes))) {
      return failure();
    }
  }

  // Check if the inferred type is valid.
  if (inferredTypes.size() != op->getNumResults()) {
    llvm::outs() << "Inferred type size does not match the number of results."
                 << "inferredTypes.size() = " << inferredTypes.size()
                 << ", op->getNumResults() = " << op->getNumResults() << "\n";
    assert(false);
  }
  for (auto &type : inferredTypes) {
    if (!type) {
      return failure();
    }
  }

  // Set the inferred result types.
  for (unsigned i = 0; i < op->getNumResults(); i++) {
    op->getResult(i).setType(inferredTypes[i]);
  }
  return success();
}

bool verifyDefsAreUsed(Block *block) {
  mlir::DenseMap<mlir::Value, bool> values;

  for (auto &arg : block->getArguments()) {
    values[arg] = false;
  }
  for (auto &op : block->getOperations()) {
    for (auto result : op.getResults()) {
      values[result] = false;
    }
    for (auto operand : op.getOperands()) {
      values[operand] = true;
    }
  }

  for (auto &value : values) {
    if (!value.second) {
      return false;
    }
  }
  return true;
}

bool hasTargetShape(Operation *op, ArrayRef<int64_t> targetShape) {
  auto shape = op->getResult(0).getType().cast<ShapedType>().getShape();
  return shape == targetShape;
}

bool hasRankedAndKnownShape(Operation *op) {
  auto shapedType = op->getResult(0).getType().cast<ShapedType>();
  return shapedType.hasStaticShape();
}

Operation *createOpAndCandidate(SynthesisContextPtr &sCtx,
                                SynthesisResultPtr &synthesisResult,
                                ArgTuple &candidateTuple,
                                RegisteredOperationName &opName,
                                grammar::GrammarOpPtr &opInfo) {
  auto builder = OpBuilder(&sCtx->ctx);

  auto newCandidate = std::make_shared<Candidate>(
      candidateTuple.operands, grammar::OpAndResType::HLO_Tensor);
  synthesisResult->candidate = newCandidate;

  // Set up operands.
  SmallVector<mlir::Value> operands =
      newCandidate->merge(sCtx->ctx, candidateTuple.operands,
                          sCtx->options.skipMergeCandidateArguments);

  // Set up attributes.
  SmallVector<NamedAttribute> attributes = {};
  for (unsigned i = 0; i < opInfo->getNumAttributes(); i++) {
    if (!opInfo->isAttributeRequired(i))
      continue;

    std::string attrName = opInfo->getAttributeName(i);
    mlir::Attribute value = candidateTuple.attributes[i];
    attributes.push_back(builder.getNamedAttr(attrName, value));
  }

  // Set up regions.
  SmallVector<std::unique_ptr<Region>> regions = {};
  for (auto &regionCandidate : candidateTuple.regions) {
    std::unique_ptr<Region> region = std::make_unique<Region>();
    BlockAndValueMapping mapping;
    regionCandidate->cloneInto(region.get(), mapping);
    regions.push_back(std::move(region));
  }

  // Set up results types.
  // TODO: Parse number of results from the op definition.
  SmallVector<mlir::Type> resultTypes;
  if (operands.empty()) {
    resultTypes.push_back(builder.getNoneType());
  } else {
    auto opInfo = grammar::createGrammarOp(opName.getStringRef().str());
    for (unsigned resultIdx = 0; resultIdx < opInfo->getNumResults();
         resultIdx++) {
      auto opResultType = opInfo->getResultType(resultIdx);

      if (opResultType == opInfo->getOperandType(0)) {
        resultTypes.push_back(operands[resultIdx].getType());
      } else if (opResultType == grammar::OpAndResType::HLO_PredTensor) {
        auto shape =
            operands[resultIdx].getType().cast<ShapedType>().getShape();
        resultTypes.push_back(
            RankedTensorType::get(shape, builder.getIntegerType(1)));
      } else {
        resultTypes.push_back(operands[resultIdx].getType());
      }
    }
  }

  // Create operation.
  Operation *op;
  if (opName.getIdentifier() == "linalg.matmul") {
    ValueRange ins = {operands[0], operands[1]};
    ValueRange outs = {operands[2]};
    op = builder.create<linalg::MatmulOp>(UnknownLoc::get(&sCtx->ctx), ins,
                                          outs);
  } else if (opName.getIdentifier() == "linalg.matvec") {
    ValueRange ins = {operands[0], operands[1]};
    ValueRange outs = {operands[2]};
    op = builder.create<linalg::MatvecOp>(UnknownLoc::get(&sCtx->ctx), ins,
                                          outs);
  } else {
    op = builder.create(UnknownLoc::get(&sCtx->ctx), opName.getIdentifier(),
                        operands, resultTypes, attributes, {}, regions);
  }

  newCandidate->addOperation(sCtx->ctx, op);

  return op;
}

SuccessOr<ProcessingStatus> runStaticChecks(SynthesisContextPtr &sCtx,
                                            SynthesisResultPtr &synthesisResult,
                                            RegisteredOperationName &opName,
                                            Operation *op) {
  auto newCandidate = synthesisResult->candidate;

  // Check length.
  if (newCandidate->getNumOps() > sCtx->options.maxNumOps) {
    return reject_hasTooManyOps;
  }

  // Infer the operation result type.
  if (failed(verifyOp(op, opName))) {
    return reject_isNotVerifiable;
  }

  if (!sCtx->options.skipTypeInference) {
    if (failed(inferResultTypes(sCtx->ctx, op))) {
      return reject_isNotResultTypeInferrable;
    }
  }

  if (failed(verifyOp(op, opName))) {
    return reject_isNotVerifiable;
  }

  // Check if the operation result shape rank is supported.
  for (auto resultType : op->getResultTypes()) {
    if (resultType.isa<RankedTensorType>()) {
      auto shape = resultType.cast<RankedTensorType>().getShape();
      if (shape.size() > maxShapeRank) {
        return reject_hasUnsupportedShapeRank;
      }
    }
  }

  synthesisResult->candidate = newCandidate;
  synthesisResult->module = createModule(sCtx->ctx, newCandidate);

  if (!verifyDefsAreUsed(
          &synthesisResult->module->getRegion().getBlocks().front())) {
    return reject_isNotAllDefsAreUsed;
  }
  if (!succeeded(verify(*synthesisResult->module))) {
    return reject_isNotVerifiable;
  }
  if (newCandidate->getNumArguments() == 0) {
    return reject_hasNoArguments;
  }
  if (!hasRankedAndKnownShape(op)) {
    return reject_hasUnknownRankAndShape;
  }

  sCtx->stats.numValid++;
  auto returnShape = op->getResult(0).getType().cast<ShapedType>().getShape();
  if (llvm::any_of(returnShape, [](int64_t dim) { return dim == 0; })) {
    return reject_hasZeroShape;
  }

  return success();
}

SuccessOr<ProcessingStatus> runExecution(SynthesisContextPtr &sCtx,
                                         SynthesisResultPtr &synthesisResult,
                                         Operation *op, SpecPtr &spec,
                                         CandidateStorePtr &localCandidateStore,
                                         ArrayRef<int64_t> &targetShape) {
  ModuleOp moduleCopy =
      copyModuleToCtx(&sCtx->ctx, synthesisResult->module.get());
  if (failed(sCtx->executor->lowerCHLOToLLVMDialect(moduleCopy))) {
    return reject_isNotCompilableToLLVM;
  }

  auto newCandidate = synthesisResult->candidate;

  if (!spec) {
    localCandidateStore->addCandidate(newCandidate);
  } else {
    auto func = synthesisResult->module->lookupSymbol<func::FuncOp>("foo");
    auto returnType = getReturnType(func);
    auto retCand =
        getOwningMemRefForShape(returnType.cast<ShapedType>().getShape());

    // Create args array.
    auto argsCand = selectArgs(spec->inputs, newCandidate->getArgIds());
    if (sCtx->options.withCopyArgs)
      argsCand = copyArgs(argsCand);

    // Compile and run.
    if (failed(jitAndInvoke(moduleCopy, argsCand, retCand)))
      return reject_isNotExecutable;
    sCtx->stats.numExecuted++;

    double *out = getReturnDataPtr(retCand);

    // Hash and add to store if hash doesn't exist yet.
    auto returnShape = op->getResult(0).getType().cast<ShapedType>().getShape();
    double hash = hashArray(out, returnShape);
    newCandidate->setHash(hash);

    if (sCtx->options.ignoreEquivalentCandidates &&
        !sCtx->candidateStore->addCandidateHash(hash)) {
      sCtx->stats.numIgnored++;
      return reject_hashNotUnique;
    }

    localCandidateStore->addCandidate(newCandidate);

    if (returnShape == targetShape) {
      double *refOut = getReturnDataPtr(spec->output);
      if (areArraysEqual(refOut, out, returnShape)) {
        LLVM_DEBUG(llvm::dbgs() << "Found a match!\n");
        LLVM_DEBUG(synthesisResult->module->print(llvm::dbgs()));

        sCtx->candidateStore->merge(localCandidateStore);
        sCtx->stats.numOps = newCandidate->getNumOps();

        return accept_as_solution;
      }
    }
  }

  return success();
}

ProcessingStatus
processCandidate(SynthesisContextPtr &sCtx, RegisteredOperationName &opName,
                 grammar::GrammarOpPtr &opInfo, 
                 CandidateStorePtr &localCandidateStore,
                 ArgTuple candidateTuple, SynthesisResultPtr &synthesisResult,
                 mlir::Type returnType,
                 SpecPtr spec = nullptr) {
  sCtx->stats.numSynthesized++;

  synthesisResult = std::make_shared<SynthesisResult>();

  Operation *op = createOpAndCandidate(sCtx, synthesisResult, candidateTuple,
                                       opName, opInfo);

  auto staticCheckResult = runStaticChecks(sCtx, synthesisResult, opName, op);
  if (failed(staticCheckResult)) {
    return staticCheckResult.value();
  }

  auto targetShape = returnType.cast<ShapedType>().getShape();
  auto executionResult =
      runExecution(sCtx, synthesisResult, op, spec, localCandidateStore,
                   targetShape);
  if (failed(executionResult)) {
    return executionResult.value();
  } else {
    return accept_as_candidate;
  }
}

SpecPtr generateSpec(MLIRContext &ctx, IExecutorPtr &executor,
                     func::FuncOp inputFunction) {
  auto inputFunctionName = inputFunction.getName().str();
  auto targetShape = getReturnType(inputFunction).cast<ShapedType>().getShape();
  prepareInputFunction(inputFunction);

  // Compile and run reference.
  // - Create argument vector.
  auto args = createArgs(inputFunction);
  randomlyInitializeArgs(inputFunction, args);
  auto ret = getOwningMemRefForShape(targetShape);

  // - Run on argument vector gives the reference out.
  auto inputModuleRef = createModule(ctx, &inputFunction);
  auto inputModule = inputModuleRef.release();
  assert(succeeded(executor->lowerAffineToLLVMDialect(inputModule)));
  assert(succeeded(jitAndInvoke(inputModule, args, ret)));

  convertScalarToMemrefArgs(args);

  return std::make_shared<Spec>(args, ret);
}

std::tuple<operandsSetTy, attributesSetTy, regionsSetTy>
getOperandsAttributesRegions(SynthesisContextPtr &sCtx,
                             RegisteredOperationName opName,
                             std::vector<mlir::Type> &inputFunctionArgTypes,
                             mlir::Type returnType, int numOps) {
  operandsSetTy operands;
  attributesSetTy attributes;
  regionsSetTy regions;

  // - Operands.
  auto opInfo = grammar::createGrammarOp(opName.getStringRef().str());
  for (unsigned i = 0; i < opInfo->getNumOperands(); i++) {
    std::vector<CandidatePtr> operandCandidates =
        sCtx->options.ignoreTypes ? sCtx->candidateStore->getCandidates(numOps)
                                  : sCtx->candidateStore->getCandidates(
                                        numOps, opInfo->getOperandType(i));
    operands.push_back(operandCandidates);
  }

  // - Attributes.
  auto attrGen = std::make_shared<AttributeGenerator>(
      sCtx->ctx, inputFunctionArgTypes, returnType);
  attributes = opInfo->genAttributes(attrGen);

  // - Regions.
  auto regionsGenereated = genRegions(sCtx->ctx);
  for (unsigned i = 0; i < opInfo->getNumRegions(); i++) {
    regions.push_back(regionsGenereated);
  }

  return std::make_tuple(operands, attributes, regions);
}

std::vector<RegisteredOperationName>
getDialectOps(MLIRContext *ctx, std::vector<Dialect *> &dialects,
              const std::vector<std::string> &ops, bool printOps) {
  std::unordered_map<std::string, RegisteredOperationName>
      opNamesToRegisteredOps;
  for (RegisteredOperationName op : ctx->getRegisteredOperations()) {
    std::string opStr = op.getIdentifier().str();
    opNamesToRegisteredOps.insert({opStr, op});
  }

  llvm::SetVector<RegisteredOperationName> selectedRegisteredOps;
  for (auto &op : ops) {
    std::string opStr = std::string(op);
    // selectedRegisteredOps.insert(opNamesToRegisteredOps[opStr]);
    auto it = opNamesToRegisteredOps.find(opStr);
    if (it != opNamesToRegisteredOps.end()) {
      selectedRegisteredOps.insert(it->second);
    } else {
      assert(false && "Op not found in registered ops");
    }
  }

  if (printOps) {
    llvm::outs() << "Registered ops:"
                 << "\n--------\n";
    for (auto opName : selectedRegisteredOps.takeVector()) {
      opName.dump();
      llvm::outs() << "\n";
    }
  }

  return selectedRegisteredOps.takeVector();
}

SynthesisResultPtr
synthesize(SynthesisContextPtr &sCtx, func::FuncOp inputFunction,
           std::vector<mlir::RegisteredOperationName> avaliableOps) {
  std::vector<mlir::Type> argTypes = inputFunction.getArgumentTypes();
  auto returnType = getReturnType(inputFunction);

  // Generate spec.
  auto spec = generateSpec(sCtx->ctx, sCtx->executor, inputFunction);
  if (sCtx->options.printArgsAndResults)
    spec->dumpAsPython();

  // Init candidate store.
  auto candidates = sCtx->initialCandidateGen->gen(argTypes, returnType);
  for (auto &candidate : candidates)
    sCtx->candidateStore->addCandidate(candidate);

  // Synthesize.
  SynthesisResultPtr result;
  auto synthStart = std::chrono::high_resolution_clock::now();

  for (int numOps = 0; numOps <= sCtx->options.maxNumOps; numOps++) {
    CandidateStorePtr localCandidateStore = std::make_shared<CandidateStore>();

    for (auto opName : avaliableOps) {
      auto opInfo = grammar::createGrammarOp(opName.getStringRef().str());

      // Build cartesian product of operation operands, attributes and regions.
      auto [operands, attributes, regions] = getOperandsAttributesRegions(
          sCtx, opName, argTypes, returnType, numOps);
      CartesianProduct cartesianProduct(sCtx->options.maxNumOps);
      auto candidateTuples =
          cartesianProduct.generate(operands, attributes, regions);

      std::string statusStr;
      if (sCtx->options.printSynthesisSteps) {
        statusStr = "Level: " + std::to_string(numOps) +
                    ", op: " + opName.getStringRef().str() +
                    ", candidates: " + std::to_string(candidateTuples.size());
      }

      // Check each candidate in the cartesian product.
      auto status = failableParallelForEachWithProgress(
          &sCtx->ctx, candidateTuples, statusStr, [&](auto &candidateTuple) {
            // Check if timeout.
            auto synthEnd = std::chrono::high_resolution_clock::now();
            auto synthDuration =
                std::chrono::duration_cast<std::chrono::seconds>(synthEnd -
                                                                 synthStart)
                    .count();
            if (sCtx->options.timeoutPerFunction &&
                synthDuration > sCtx->options.timeoutPerFunction)
              return failure();

            // Process candidate.
            SynthesisResultPtr synthesisResult;
            SynthesisStats synthesisStats;
            ProcessingStatus status = processCandidate(
                sCtx, opName, opInfo, localCandidateStore, candidateTuple,
                synthesisResult, returnType, spec);
            synthesisStats.addProcessingStatus(status);
            sCtx->stats.merge(synthesisStats);

            // Print candidate.
            printCandidate(status, localCandidateStore, sCtx->candidateStore,
                           sCtx->options, synthesisResult, sCtx->printMutex);

            // Check if solution.
            if (status == accept_as_solution) {
              result = synthesisResult;
              finalizeFunction(
                  result->module->lookupSymbol<func::FuncOp>("foo"),
                  inputFunction.getName().str());

              if (sCtx->options.stopOnSolutionCandidate)
                return failure();
            }

            return success();
          });
      if (failed(status)) {
        if (result)
          return result;
        return result;
      }
    }

    sCtx->candidateStore->merge(localCandidateStore);
  }

  return result;
}

void 
enumerateOneOp(SynthesisContextPtr &sCtx, std::vector<mlir::Type> argTypes, mlir::Type returnType,
               std::vector<mlir::RegisteredOperationName> avaliableOps) {
  // Init candidate store.
  auto candidates = sCtx->initialCandidateGen->gen(argTypes, returnType);
  for (auto &candidate : candidates) {
    sCtx->candidateStore->addCandidate(candidate);
  }

  // Synthesize.
  CandidateStorePtr localCandidateStore = std::make_shared<CandidateStore>();

  int numOps = 2;
  for (auto opName : avaliableOps) {
    auto opInfo = grammar::createGrammarOp(opName.getStringRef().str());

    // Build cartesian product of operation operands, attributes and regions.
    auto [operands, attributes, regions] = getOperandsAttributesRegions(
        sCtx, opName, argTypes, returnType, numOps);
    CartesianProduct cartesianProduct(2);
    auto candidateTuples =
        cartesianProduct.generate(operands, attributes, regions);

    // Check each candidate in the cartesian product.
    std::string statusStr;
    auto status = failableParallelForEachWithProgress(
        &sCtx->ctx, candidateTuples, statusStr, [&](auto &candidateTuple) {
          // Process candidate.
          SynthesisResultPtr synthesisResult;
          SynthesisStats synthesisStats;
          ProcessingStatus status = processCandidate(
              sCtx, opName, opInfo, localCandidateStore, candidateTuple,
              synthesisResult, returnType);
          synthesisStats.addProcessingStatus(status);
          sCtx->stats.merge(synthesisStats);

          // Print candidate.
          printCandidate(status, localCandidateStore, sCtx->candidateStore,
                         sCtx->options, synthesisResult, sCtx->printMutex);

          return success();
        });
  }

  sCtx->candidateStore->merge(localCandidateStore);
}