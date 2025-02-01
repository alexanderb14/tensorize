#include "ProcessingStatus.h"

std::string processingStatusToStr(ProcessingStatus &status) {
  if (status == reject_hasTooManyOps)
    return "reject_hasTooManyOps";
  if (status == reject_isNotResultTypeInferrable)
    return "reject_isNotResultTypeInferrable";
  if (status == reject_hasUnsupportedShapeRank)
    return "reject_hasUnsupportedShapeRank";
  if (status == reject_isNotAllDefsAreUsed)
    return "reject_isNotAllDefsAreUsed";
  if (status == reject_isNotVerifiable)
    return "reject_isNotVerifiable";
  if (status == reject_hasNoArguments)
    return "reject_hasNoArguments";
  if (status == reject_hasUnknownRankAndShape)
    return "reject_hasUnknownRankAndShape";
  if (status == reject_hasZeroShape)
    return "reject_hasZeroShape";
  if (status == reject_isNotCompilableToLLVM)
    return "reject_isNotCompilableToLLVM";
  if (status == reject_isNotExecutable)
    return "reject_isNotExecutable";
  if (status == reject_hashNotUnique)
    return "reject_hashNotUnique";
  if (status == accept_as_candidate)
    return "accept_as_candidate";
  if (status == accept_as_solution)
    return "accept_as_solution";
  assert(false && "Processing Status not known");
}
