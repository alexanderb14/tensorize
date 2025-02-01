#include "CandidateStore.h"
#include "synthesis/Generators.h"

using namespace llvm;
using namespace mlir;

void CandidateStore::addCandidate(const CandidatePtr &candidate) {
  std::lock_guard<std::mutex> lock(addCandidatesMutex);

  unsigned weight = candidate->getNumOps();
  grammar::OpAndResType opAndResType = candidate->getOpAndResType();
  opAndResType = getTypeAlias(opAndResType);

  candidateToId[candidate.get()] = candidateToId.size();

  if (weightToCandidates.find(weight) == weightToCandidates.end())
    weightToCandidates[weight] =
        std::unordered_map<grammar::OpAndResType, std::vector<CandidatePtr>>();

  if (weightToCandidates[weight].find(opAndResType) ==
      weightToCandidates[weight].end())
    weightToCandidates[weight][opAndResType] = std::vector<CandidatePtr>();

  weightToCandidates[weight][opAndResType].push_back(candidate);
}

std::vector<CandidatePtr> CandidateStore::getCandidates() {
  std::vector<CandidatePtr> candidates;
  for (auto &weightToCandidate : weightToCandidates) {
    for (auto &opAndResTypeToCandidate : weightToCandidate.second) {
      for (auto &candidate : opAndResTypeToCandidate.second) {
        candidates.push_back(candidate);
      }
    }
  }
  return candidates;
}

std::vector<CandidatePtr> CandidateStore::getCandidates(unsigned weight) {
  std::vector<CandidatePtr> candidates;
  for (unsigned i = 0; i < weight; i++) {
    if (weightToCandidates.find(i) != weightToCandidates.end()) {
      for (auto &opAndResTypeToCandidate : weightToCandidates[i]) {
        for (auto &candidate : opAndResTypeToCandidate.second) {
          candidates.push_back(candidate);
        }
      }
    }
  }
  return candidates;
}

std::vector<CandidatePtr>
CandidateStore::getCandidates(unsigned weight,
                              grammar::OpAndResType opAndResType) {
  opAndResType = getTypeAlias(opAndResType);

  std::vector<CandidatePtr> candidates;

  for (unsigned i = 0; i < weight; i++) {
    if (weightToCandidates.find(i) != weightToCandidates.end()) {
      if (weightToCandidates[i].find(opAndResType) !=
          weightToCandidates[i].end()) {
        for (auto &candidate : weightToCandidates[i][opAndResType]) {
          candidates.push_back(candidate);
        }
      }
    }
  }
  return candidates;
}

void CandidateStore::merge(CandidateStorePtr &other) {
  for (auto &pair : other->weightToCandidates) {
    for (auto &opAndResTypeToCandidate : pair.second) {
      for (auto &candidate : opAndResTypeToCandidate.second) {
        addCandidate(candidate);
      }
    }
  }
}

void CandidateStore::dumpCandidates() {
  for (auto &pair : weightToCandidates) {
    llvm::outs() << "Weight: " << pair.first << "\n";
    for (auto &opAndResTypeToCandidate : pair.second) {
      llvm::outs() << "OpAndResType: "
                   << opAndResTypeToString(opAndResTypeToCandidate.first)
                   << "\n";
      for (auto &candidate : opAndResTypeToCandidate.second) {
        candidate->dump();
      }
    }
  }
}

void CandidateStore::dumpSizes() {
  llvm::outs() << "\nCandidateStore contents (length: number of candidates)"
               << "\n--------\n";

  for (auto &pair : weightToCandidates) {
    for (auto &opAndResTypeToCandidate : pair.second) {
      llvm::outs() << pair.first << ": "
                   << opAndResTypeToString(opAndResTypeToCandidate.first)
                   << ": " << opAndResTypeToCandidate.second.size() << "\n";
    }
  }
}

int CandidateStore::getTotal() {
  int numCandidates = 0;
  for (auto &pair : weightToCandidates) {
    numCandidates += pair.second.size();
  }
  return numCandidates;
}

int CandidateStore::getCandidateId(const CandidatePtr &candidate) {
  auto pos = candidateToId.find(candidate.get());
  if (pos == candidateToId.end())
    return -1;
  return pos->second;
}

bool CandidateStore::addCandidateHash(double hash) {
  std::lock_guard<std::mutex> lock(hashesMutex);

  if (hashes.find(hash) == hashes.end()) {
    hashes[hash] = 1;
    return true;
  }

  hashes[hash]++;
  return false;
}
