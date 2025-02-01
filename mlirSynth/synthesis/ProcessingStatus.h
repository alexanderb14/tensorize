#ifndef IRSYNTH_PROCESSINGSTATUS_H
#define IRSYNTH_PROCESSINGSTATUS_H

#include <string>
#include <assert.h>

enum ProcessingStatus {
  reject_hasTooManyOps,
  reject_isNotResultTypeInferrable,
  reject_hasUnsupportedShapeRank,
  reject_isNotAllDefsAreUsed,
  reject_isNotVerifiable,
  reject_hasNoArguments,
  reject_hasUnknownRankAndShape,
  reject_hasZeroShape,
  reject_isNotCompilableToLLVM,
  reject_isNotExecutable,
  reject_hashNotUnique,
  accept_as_candidate,
  accept_as_solution,
  ENUM_SIZE,
};

std::string processingStatusToStr(ProcessingStatus &status);

#endif // IRSYNTH_PROCESSINGSTATUS_H
