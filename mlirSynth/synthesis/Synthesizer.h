#ifndef IRSYNTH_SYNTHESIZER_H
#define IRSYNTH_SYNTHESIZER_H

#include "execution/Executor.h"
#include "synthesis/CandidateStore.h"
#include "synthesis/Generators.h"
#include "synthesis/Options.h"
#include "synthesis/Stats.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"

#include <vector>

struct SynthesisContext {
  SynthesisContext(mlir::MLIRContext &ctx, IExecutorPtr executor,
                   InitialCandidateGeneratorPtr initialCandidateGen,
                   CandidateStorePtr candidateStore, SynthesisOptions &options,
                   SynthesisStats &stats)
      : ctx(ctx), executor(executor), initialCandidateGen(initialCandidateGen),
        candidateStore(candidateStore), options(options), stats(stats) {}

  mlir::MLIRContext &ctx;
  IExecutorPtr executor;
  InitialCandidateGeneratorPtr initialCandidateGen;
  CandidateStorePtr candidateStore;
  SynthesisOptions &options;
  SynthesisStats &stats;
  std::mutex printMutex;
};
using SynthesisContextPtr = std::shared_ptr<SynthesisContext>;

struct SynthesisResult {
  CandidatePtr candidate;
  mlir::OwningOpRef<mlir::ModuleOp> module;
};
using SynthesisResultPtr = std::shared_ptr<SynthesisResult>;

std::vector<mlir::RegisteredOperationName>
getDialectOps(mlir::MLIRContext *ctx, std::vector<mlir::Dialect *> &dialects,
              const std::vector<std::string> &ops, bool printOps);

void initializeCandidates(mlir::MLIRContext &ctx,
                          CandidateStorePtr &candidateStore,
                          mlir::Region::BlockArgListType functionArgs,
                          llvm::ArrayRef<int64_t> targetShape);

mlir::OwningOpRef<mlir::func::FuncOp> unwrapModule(mlir::ModuleOp &module);

mlir::OwningOpRef<mlir::ModuleOp> createModule(mlir::MLIRContext &ctx,
                                               mlir::func::FuncOp *function);
mlir::OwningOpRef<mlir::ModuleOp> createModule(mlir::MLIRContext &ctx,
                                               CandidatePtr candidate);

SynthesisResultPtr
synthesize(SynthesisContextPtr &sCtx, mlir::func::FuncOp inputFunction,
           std::vector<mlir::RegisteredOperationName> avaliableOps);

void
enumerateOneOp(SynthesisContextPtr &sCtx, std::vector<mlir::Type> argTypes, mlir::Type returnType,
               std::vector<mlir::RegisteredOperationName> avaliableOps);

#endif // IRSYNTH_SYNTHESIZER_H
