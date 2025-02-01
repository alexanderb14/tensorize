#include "Stats.h"

#include "llvm/Support/raw_ostream.h"

SynthesisStats::SynthesisStats() {
  // Initialize all the processing status counts with 0.
  for (int i = 0; i < static_cast<int>(ProcessingStatus::ENUM_SIZE); i++) {
    processingStatusCounts.insert({(ProcessingStatus)i, 0});
  }
}

void SynthesisStats::addProcessingStatus(ProcessingStatus status) {
  processingStatusCounts[status]++;
}

void SynthesisStats::merge(SynthesisStats &other) {
  std::lock_guard<std::mutex> lock(mutex);

  numSynthesized += other.numSynthesized;
  numValid += other.numValid;
  numExecuted += other.numExecuted;
  numIgnored += other.numIgnored;
  numOps += other.numOps;

  for (auto &kv : other.processingStatusCounts) {
    processingStatusCounts[kv.first] += kv.second;
  }
}

void SynthesisStats::dump() {
  llvm::outs() << "Synthesis Stats"
               << "\n--------\n";
  llvm::outs() << "Number of synthesized candidates:             "
               << numSynthesized << "\n";

  llvm::outs() << "Number of valid candidates:                  " << numValid
               << "\n";
  llvm::outs() << "Percentage of valid candidates:              "
               << (numValid * 100.0) / numSynthesized << "%\n";

  llvm::outs() << "Number of executed candidates:               " << numExecuted
               << "\n";
  llvm::outs() << "Percentage of executed candidates:           "
               << (numExecuted * 100.0) / numSynthesized << "%\n";

  llvm::outs() << "Number of ignored equivalent candidates:     " << numIgnored
               << "\n";

  llvm::outs() << "Number of operations in solution candidate:  " << numOps
               << "\n";

  llvm::outs() << "Number of operations per function:           ";
  for (unsigned numOps : numOpsPerFunction) {
    llvm::outs() << numOps << " ";
  }
  llvm::outs() << "\n";

  llvm::outs() << "Processing Statuses:\n";
  for (auto &kv : processingStatusCounts) {
    ProcessingStatus status = kv.first;
    llvm::outs() << "  " << processingStatusToStr(status) << ": " << kv.second
                 << "\n";
  }

  llvm::outs() << "JSON: {";

  llvm::outs() << "\"numSynthesized\":" << numSynthesized << ","
               << "\"numValid\":" << numValid << ","
               << "\"numExecuted\":" << numExecuted << ","
               << "\"numIgnored\":" << numIgnored << ","
               << "\"numOps\":" << numOps << ",";

  llvm::outs() << "\"numOpsPerFunction\":[";
  bool first2 = true;
  for (unsigned numOps : numOpsPerFunction) {
    if (!first2) {
      llvm::outs() << ",";
    }
    first2 = false;
    llvm::outs() << numOps;
  }
  llvm::outs() << "],";

  llvm::outs() << "\"processingStatusCounts\":{";
  bool first = true;
  for (auto &kv : processingStatusCounts) {
    if (!first) {
      llvm::outs() << ",";
    }
    first = false;
    ProcessingStatus status = kv.first;
    llvm::outs() << "\"" << processingStatusToStr(status) << "\":" << kv.second;
  }
  llvm::outs() << "}";

  llvm::outs() << "}\n";
}
