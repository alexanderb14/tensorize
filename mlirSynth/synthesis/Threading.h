#ifndef THREADING_H
#define THREADING_H

#include "mlir/IR/Diagnostics.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/Support/ThreadPool.h"

#include "indicators/indicators.hpp"

#include <atomic>

using namespace mlir;
using namespace indicators;

/// Invoke the given function on the elements between [begin, end)
/// asynchronously. If the given function returns a failure when processing any
/// of the elements, execution is stopped and a failure is returned from this
/// function. This means that in the case of failure, not all elements of the
/// range will be processed. Diagnostics emitted during processing are ordered
/// relative to the element's position within [begin, end). If the provided
/// context does not have multi-threading enabled, this function always
/// processes elements sequentially.
template <typename RangeT, typename FuncT>
LogicalResult failableParallelForEachWithProgress(MLIRContext *context, RangeT &&range, std::string statusStr,
                                      FuncT &&func) {
  auto begin = std::begin(range);
  auto end = std::end(range);
                                        
  unsigned numElements = static_cast<unsigned>(std::distance(begin, end));
  if (numElements == 0)
    return success();

  // If multithreading is disabled or there is a small number of elements,
  // process the elements directly on this thread.
  if (!context->isMultithreadingEnabled() || numElements <= 1) {
    for (; begin != end; ++begin)
      if (failed(func(*begin)))
        return failure();
    return success();
  }

  ProgressBar bar{
      option::BarWidth{20},
      option::PostfixText{statusStr},
      option::Fill{"■"},
      option::Lead{"■"},
      option::ShowElapsedTime{true},
      option::ShowRemainingTime{true},
      option::MaxProgress{numElements}
  };

  // Build a wrapper processing function that properly initializes a parallel
  // diagnostic handler.
  ParallelDiagnosticHandler handler(context);
  std::atomic<unsigned> curIndex(0);
  std::atomic<bool> processingFailed(false);

  auto processFn = [&] {
    while (!processingFailed) {
      unsigned index = curIndex++;
      if (index >= numElements)
        break;

      if (!statusStr.empty()) {
        bar.tick();
      }
    
      handler.setOrderIDForThread(index);
      if (failed(func(*std::next(begin, index))))
        processingFailed = true;
      handler.eraseOrderIDForThread();
    }
  };

  // Otherwise, process the elements in parallel.
  llvm::ThreadPool &threadPool = context->getThreadPool();
  llvm::ThreadPoolTaskGroup tasksGroup(threadPool);
  size_t numActions = std::min(numElements, threadPool.getThreadCount());
  for (unsigned i = 0; i < numActions; ++i) {
    tasksGroup.async(processFn);
  }
  // If the current thread is a worker thread from the pool, then waiting for
  // the task group allows the current thread to also participate in processing
  // tasks from the group, which avoid any deadlock/starvation.
  tasksGroup.wait();
  return failure(processingFailed);
}

#endif // THREADING_H