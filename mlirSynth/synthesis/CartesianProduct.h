#ifndef IRSYNTH_CANDIDATETUPLES_H
#define IRSYNTH_CANDIDATETUPLES_H

#include "Candidate.h"

struct ArgTuple {
  std::vector<CandidatePtr> operands;
  std::vector<mlir::Attribute> attributes;
  std::vector<std::shared_ptr<mlir::Region>> regions;
};

using RegionPtr = std::shared_ptr<mlir::Region>;

class CartesianProduct {
public:
  CartesianProduct(int maxNumOps) { this->maxNumOps = maxNumOps; }

  std::vector<ArgTuple>
  generate(std::vector<std::vector<CandidatePtr>> &operands,
           std::vector<std::vector<mlir::Attribute>> &attributes,
           std::vector<std::vector<RegionPtr>> &regions);

private:
  int maxNumOps;
};

#endif // IRSYNTH_CANDIDATETUPLES_H
