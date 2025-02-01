#ifndef IRSYNTH_CANDIDATE_H
#define IRSYNTH_CANDIDATE_H

#include "synthesis/Grammar.h"

#include "mlir/IR/Block.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Region.h"

#include <mutex>

class Candidate;
using CandidatePtr = std::shared_ptr<Candidate>;

class Candidate {
public:
  Candidate(std::vector<CandidatePtr> predecessors,
            grammar::OpAndResType returnType) {
    this->predecessors = predecessors;
    this->returnType = returnType;
    region = std::make_shared<mlir::Region>();
    region->push_back(new mlir::Block);
  }

  mlir::SmallVector<mlir::Value>
  merge(mlir::MLIRContext &ctx, std::vector<CandidatePtr> &others,
        bool skipMergeCandidateArguments = false);

  void addArgument(mlir::MLIRContext &ctx, mlir::Type type, unsigned argId);
  void addOperation(mlir::MLIRContext &ctx, mlir::Operation *op,
                    bool count = true);

  mlir::Operation *getBegin() { return &region->getBlocks().front().front(); }
  mlir::Region *getRegion() { return region.get(); }
  std::vector<unsigned> getArgIds() { return argIds; }

  mlir::Operation *getEnd() { return &this->getBlock()->back(); }
  mlir::Block *getBlock() { return &region->getBlocks().front(); }

  mlir::SmallVector<mlir::Value> getResults();
  int getNumOps() { return numOps; }
  int getNumArguments() {
    return region->getBlocks().front().getNumArguments();
  }
  void dump();

  std::vector<CandidatePtr> getPredecessors() { return predecessors; }

  bool hasHash() { return hashExists; }
  void setHash(double hash) {
    hashExists = true;
    this->hash = hash;
  }
  double getHash() { return hash; }

  void setOpAndResType(grammar::OpAndResType ioType) {
    this->returnType = ioType;
  }
  grammar::OpAndResType getOpAndResType() { return returnType; }

private:
  std::vector<unsigned> argIds;
  std::shared_ptr<mlir::Region> region;
  int numOps = 0;

  std::vector<CandidatePtr> predecessors;

  double hash;
  bool hashExists = false;

  grammar::OpAndResType returnType;
};

#endif // IRSYNTH_CANDIDATE_H
