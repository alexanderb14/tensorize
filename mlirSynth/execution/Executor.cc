#include "Executor.h"

#include "execution/ArgUtils.h"
#include "execution/Lowering.h"

#include "mlir/Bytecode/BytecodeReader.h"
#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/ExecutionEngine/CRunnerUtils.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/MemRefUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LLVM.h"
#include "llvm/Support/JSON.h"
#include <chrono>

using namespace mlir;

ModuleOp copyModuleToCtx(MLIRContext *ctx, ModuleOp module) {
  // Export the module using old context.
  std::string oss;
  llvm::raw_string_ostream os(oss);
  writeBytecodeToFile(module, os);

  // Import the module using new context.
  llvm::MemoryBufferRef bufferRef(os.str(), "bytecode");
  Block parsedBlock;
  ParserConfig config(ctx, false);
  if (failed(readBytecodeFile(bufferRef, &parsedBlock, config)))
    assert(false && "Failed to import the module");

  // Create a new module containing all functions from the parsed block.
  OpBuilder builder(ctx);
  ModuleOp newModule = builder.create<ModuleOp>(UnknownLoc::get(ctx));
  parsedBlock.walk(
      [&](func::FuncOp func) { newModule.push_back(func.clone()); });

  return newModule;
}

llvm::Error invokePacked(std::unique_ptr<ExecutionEngine> jit, StringRef name,
                         MutableArrayRef<void *> args) {
  return jit->invokePacked(name, args);
}

llvm::Error invokePacked(std::unique_ptr<ExecutionEngine> jit, StringRef name,
                         MutableArrayRef<void *> args,
                         std::chrono::nanoseconds *elapsedTime,
                         int numRuns = 10) {
  std::vector<std::chrono::nanoseconds> elapsedTimes;

  for (int i = 0; i < numRuns; ++i) {
    auto startTime = std::chrono::high_resolution_clock::now();
    llvm::Error error = jit->invokePacked(name, args);
    if (error)
      return error;
    auto endTime = std::chrono::high_resolution_clock::now();
    auto singleElapsedTime =
        std::chrono::duration_cast<std::chrono::nanoseconds>(endTime -
                                                             startTime);
    elapsedTimes.push_back(singleElapsedTime);
  }

  // Get median.
  std::sort(elapsedTimes.begin(), elapsedTimes.end());
  *elapsedTime = elapsedTimes[elapsedTimes.size() / 2];

  return llvm::Error::success();
}

LogicalResult jitAndInvoke(OwningOpRef<ModuleOp> module,
                           std::vector<ReturnAndArgType> &args,
                           ReturnAndArgType &ret,
                           std::chrono::nanoseconds *elapsedTime) {
  // JIT.
  std::string runnerUtilsLib = "/devel/git/mlirSynth/deps/llvm-project/build/lib/libmlir_runner_utils.so";
  std::string cRunnerUtilsLib = "/devel/git/mlirSynth/deps/llvm-project/build/lib/libmlir_c_runner_utils.so";
  ExecutionEngineOptions options;
  options.sharedLibPaths = {runnerUtilsLib, cRunnerUtilsLib};

  auto jitOrError = ExecutionEngine::create(*module, options);
  if (!jitOrError) {
    llvm::outs() << "Error creating JIT: " << jitOrError.takeError() << "\n";
    llvm::outs() << "Module:\n";
    module->dump();
    return failure();
  }
  std::unique_ptr<ExecutionEngine> jit = std::move(jitOrError.get());

  // Prepare return and args.
  llvm::SmallVector<void *> argsArray;

  // Return
  if (auto *memRef = std::get_if<OwningMemRef0DPtr>(&ret)) {
    auto *r = new mlir::ExecutionEngine::Result<
        OwningMemRef<double, 0>::DescriptorType>(***memRef);
    argsArray.push_back(r);
  } else if (auto *memRef = std::get_if<OwningMemRef1DPtr>(&ret)) {
    auto *r = new mlir::ExecutionEngine::Result<
        OwningMemRef<double, 1>::DescriptorType>(***memRef);
    argsArray.push_back(r);
  } else if (auto *memRef = std::get_if<OwningMemRef2DPtr>(&ret)) {
    auto *r = new mlir::ExecutionEngine::Result<
        OwningMemRef<double, 2>::DescriptorType>(***memRef);
    argsArray.push_back(r);
  } else if (auto *memRef = std::get_if<OwningMemRef3DPtr>(&ret)) {
    auto *r = new mlir::ExecutionEngine::Result<
        OwningMemRef<double, 3>::DescriptorType>(***memRef);
    argsArray.push_back(r);
  } else if (auto *memRef = std::get_if<OwningMemRef4DPtr>(&ret)) {
    auto *r = new mlir::ExecutionEngine::Result<
        OwningMemRef<double, 4>::DescriptorType>(***memRef);
    argsArray.push_back(r);
  } else {
    llvm::outs() << "Unsupported return type: " << ret.index() << "\n";
    assert(false);
  }

  // Args
  llvm::SmallVector<void *> argsPtrs; // Used to hold pointers on the stack
  for (auto &returnOrArg : args) {
    if (auto *memRef = std::get_if<OwningMemRef0DPtr>(&returnOrArg)) {
      argsPtrs.push_back(&***memRef);
      argsArray.push_back(&argsPtrs.back());
    } else if (auto *memRef = std::get_if<OwningMemRef1DPtr>(&returnOrArg)) {
      argsPtrs.push_back(&***memRef);
      argsArray.push_back(&argsPtrs.back());
    } else if (auto *memRef = std::get_if<OwningMemRef2DPtr>(&returnOrArg)) {
      argsPtrs.push_back(&***memRef);
      argsArray.push_back(&argsPtrs.back());
    } else if (auto *memRef = std::get_if<OwningMemRef3DPtr>(&returnOrArg)) {
      argsPtrs.push_back(&***memRef);
      argsArray.push_back(&argsPtrs.back());
    } else if (auto *memRef = std::get_if<OwningMemRef4DPtr>(&returnOrArg)) {
      argsPtrs.push_back(&***memRef);
      argsArray.push_back(&argsPtrs.back());
    } else if (auto *arg = std::get_if<DoublePtr>(&returnOrArg)) {
      argsPtrs.push_back((void *)*arg);
      argsArray.push_back((void *)*arg);
    } else {
      assert(false && "Unsupported type");
    }
  }

  // Invoke.
  const std::string adapterName = std::string("_mlir_ciface_") + "foo";

  if (elapsedTime) {
    llvm::Error error =
        invokePacked(std::move(jit), adapterName, argsArray, elapsedTime);
    llvm::outs() << "Elapsed time: " << elapsedTime->count()
                 << " nanoseconds\n";

    if (error) {
      llvm::errs() << "Error invoking function: " << error << "\n";
      return failure();
    }
  } else {
    llvm::Error error = invokePacked(std::move(jit), adapterName, argsArray);

    if (error) {
      llvm::errs() << "Error invoking function: " << error << "\n";
      return failure();
    }
  }

  return success();
}

Executor::Executor(MLIRContext *ctx) : ctx(ctx) {
  pmCHLOToLLVM = std::make_shared<PassManager>(ctx);
  HLO::addCHLOToAffinePasses(pmCHLOToLLVM);
  HLO::addAffineToLLVMPasses(pmCHLOToLLVM);

  pmAffineToLLVM = std::make_shared<PassManager>(ctx);
  Polygeist::addAffineToLLVMPasses(pmAffineToLLVM);
}

LogicalResult Executor::lowerCHLOToLLVMDialect(ModuleOp &module) {
  std::lock_guard<std::mutex> lock(pmCHLOToLLVMMutex);
  auto result = pmCHLOToLLVM->run(module);

  return result;
}

LogicalResult Executor::lowerAffineToLLVMDialect(ModuleOp &module) {
  std::lock_guard<std::mutex> lock(pmAffineToLLVMMutex);
  auto result = pmAffineToLLVM->run(module);

  return result;
}

ThreadedExecutor::ThreadedExecutor(ContextManagerPtr ctxManager,
                                   unsigned numExecutors) {
  for (unsigned i = 0; i < numExecutors; i++) {
    MLIRContext *ctx = ctxManager->createContext();
    executors.push_back(std::make_shared<Executor>(ctx));
  }
}

LogicalResult ThreadedExecutor::lowerCHLOToLLVMDialect(ModuleOp &module) {
  executorsMutex.lock();
  auto executor = executors.back();
  executors.pop_back();
  executorsMutex.unlock();

  ModuleOp moduleCopy = copyModuleToCtx(executor->ctx, module);
  auto result = executor->lowerCHLOToLLVMDialect(moduleCopy);
  module = moduleCopy;

  executorsMutex.lock();
  executors.push_back(executor);
  executorsMutex.unlock();

  return result;
}

LogicalResult ThreadedExecutor::lowerAffineToLLVMDialect(ModuleOp &module) {
  executorsMutex.lock();
  auto executor = executors.back();
  executors.pop_back();
  executorsMutex.unlock();

  ModuleOp moduleCopy = copyModuleToCtx(executor->ctx, module);
  auto result = executor->lowerAffineToLLVMDialect(moduleCopy);
  module = moduleCopy;

  executorsMutex.lock();
  executors.push_back(executor);
  executorsMutex.unlock();

  return result;
}
