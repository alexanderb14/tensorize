#ifndef IRSYNTH_SYNTHESISOPTIONS_H
#define IRSYNTH_SYNTHESISOPTIONS_H

struct SynthesisOptions {
  bool printStatusNames = false;
  bool printStatusTiles = false;
  bool printValidCandidates = false;
  bool printInvalidCandidates = false;
  bool printStats = false;
  bool printArgsAndResults = false;
  bool printSynthesisSteps = true;
  int maxNumOps = 3;
  int timeoutPerFunction = 0;
  bool ignoreEquivalentCandidates = false;
  bool ignoreTypes = false;
  bool skipTypeInference = false;
  bool skipMergeCandidateArguments = false;
  bool stopOnSolutionCandidate = true;
  bool withCopyArgs = false;
  bool guide = false;
  bool distribute = false;
};

#endif // IRSYNTH_SYNTHESISOPTIONS_H