#ifndef IRSYNTH_SPEC_H
#define IRSYNTH_SPEC_H

#include "execution/ArgUtils.h"

#include <vector>

class Spec {
public:
  Spec(std::vector<ReturnAndArgType> inputs, ReturnAndArgType output)
      : inputs(inputs), output(output) {}

  void dumpAsPython();

public:
  std::vector<ReturnAndArgType> inputs;
  ReturnAndArgType output;
};
using SpecPtr = std::shared_ptr<Spec>;

#endif // IRSYNTH_SPEC_H
