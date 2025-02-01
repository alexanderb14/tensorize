#ifndef IRSYNTH_EXECUTOR_H
#define IRSYNTH_EXECUTOR_H

#include "common/ContextManager.h"
#include "execution/ArgUtils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/ExecutionEngine/MemRefUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/PassManager.h"

#include <mutex>
#include <variant>

mlir::ModuleOp copyModuleToCtx(mlir::MLIRContext *ctx, mlir::ModuleOp module);

mlir::LogicalResult
jitAndInvoke(mlir::OwningOpRef<mlir::ModuleOp> module,
             std::vector<ReturnAndArgType> &args, ReturnAndArgType &ret,
             std::chrono::nanoseconds *elapsedTime = nullptr);

class IExecutor {
public:
  virtual mlir::LogicalResult
  lowerCHLOToLLVMDialect(mlir::ModuleOp &module) = 0;
  virtual mlir::LogicalResult
  lowerAffineToLLVMDialect(mlir::ModuleOp &module) = 0;

  virtual ~IExecutor() = default;
};
using IExecutorPtr = std::shared_ptr<IExecutor>;

class Executor : public IExecutor {
public:
  Executor(mlir::MLIRContext *ctx);

  mlir::LogicalResult lowerCHLOToLLVMDialect(mlir::ModuleOp &module) override;
  mlir::LogicalResult lowerAffineToLLVMDialect(mlir::ModuleOp &module) override;

public:
  mlir::MLIRContext *ctx;

private:
  std::shared_ptr<mlir::PassManager> pmCHLOToLLVM;
  std::shared_ptr<mlir::PassManager> pmAffineToLLVM;

  std::mutex pmCHLOToLLVMMutex;
  std::mutex pmAffineToLLVMMutex;
};
using ExecutorPtr = std::shared_ptr<Executor>;

class ThreadedExecutor : public IExecutor {
public:
  ThreadedExecutor(ContextManagerPtr ctxManager, unsigned numExecutors);

  mlir::LogicalResult lowerCHLOToLLVMDialect(mlir::ModuleOp &module) override;
  mlir::LogicalResult lowerAffineToLLVMDialect(mlir::ModuleOp &module) override;

private:
  std::vector<ExecutorPtr> executors;
  std::mutex executorsMutex;
};
using ThreadedExecutorPtr = std::shared_ptr<ThreadedExecutor>;

#endif // IRSYNTH_EXECUTOR_H
