#ifndef IRSYNTH_GENERATORS_H
#define IRSYNTH_GENERATORS_H

#include "synthesis/Candidate.h"
#include "synthesis/Grammar.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Region.h"

// Attribute generators
// -----------------------------------------------------------------------------
std::vector<std::pair<mlir::Attribute, grammar::OpAndResType>>
genAttributes(mlir::MLIRContext &ctx, std::vector<mlir::Type> &argTypes,
              mlir::Type &returnType);

class AttributeGenerator : public grammar::AttributeGeneratorBase {
public:
  AttributeGenerator(mlir::MLIRContext &ctx, std::vector<mlir::Type> &argTypes,
                     mlir::Type &returnType)
      : grammar::AttributeGeneratorBase(ctx), argTypes(argTypes),
        returnType(returnType) {}

  std::vector<mlir::Attribute>
  genDenseIntElementsAttr(const std::string attributeName) override;
  std::vector<::llvm::SmallVector<int64_t>>
  genLlvmSmallVectorint64t(const std::string attributeName) override;

private:
  std::vector<mlir::Type> &argTypes;
  mlir::Type &returnType;
};
using AttributeGeneratorPtr = std::shared_ptr<AttributeGenerator>;

// Region generators
// -----------------------------------------------------------------------------
std::vector<std::shared_ptr<mlir::Region>> genRegions(mlir::MLIRContext &ctx);

// Initial candidate generators
// -----------------------------------------------------------------------------
class InitialCandidateGenerator {
public:
  InitialCandidateGenerator(mlir::MLIRContext &ctx) : ctx(ctx) {}
  virtual ~InitialCandidateGenerator() = default;

  virtual std::vector<CandidatePtr> gen(std::vector<mlir::Type> &argTypes,
                                        mlir::Type &returnType) = 0;

protected:
  mlir::MLIRContext &ctx;
};
using InitialCandidateGeneratorPtr = std::shared_ptr<InitialCandidateGenerator>;

class HLOInitialCandidateGenerator : public InitialCandidateGenerator {
public:
  HLOInitialCandidateGenerator(mlir::MLIRContext &ctx)
      : InitialCandidateGenerator(ctx) {}

  std::vector<CandidatePtr> gen(std::vector<mlir::Type> &argTypes,
                                mlir::Type &returnType) override;
};
using HLOInitialCandidateGeneratorPtr =
    std::shared_ptr<HLOInitialCandidateGenerator>;

class LinalgInitialCandidateGenerator : public InitialCandidateGenerator {
public:
  LinalgInitialCandidateGenerator(mlir::MLIRContext &ctx)
      : InitialCandidateGenerator(ctx) {}

  std::vector<CandidatePtr> gen(std::vector<mlir::Type> &argTypes,
                                mlir::Type &returnType) override;
};

// Type aliases
// -----------------------------------------------------------------------------
grammar::OpAndResType getTypeAlias(grammar::OpAndResType type);

#endif // IRSYNTH_GENERATORS_H
