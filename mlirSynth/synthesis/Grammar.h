/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Grammar (generated from tablegen)                                          *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#ifndef IRSYNTH_GRAMMAR_H
#define IRSYNTH_GRAMMAR_H


#include "mlir/IR/Attributes.h"
#include "mlir/IR/MLIRContext.h"

#include <memory>
#include <string>
#include <vector>

namespace grammar {
enum OpAndResType {
  AnyRankedTensor,
  AnyShaped,
  AnyTensor,
  AnyType,
  HLO_ComplexTensor,
  HLO_DimensionTensor,
  HLO_Fp32Or64Tensor,
  HLO_FpOrComplexTensor,
  HLO_FpTensor,
  HLO_IntFpOrComplexTensor,
  HLO_IntOrFpTensor,
  HLO_IntTensor,
  HLO_PredIntOrFpTensor,
  HLO_PredOrIntTensor,
  HLO_PredTensor,
  HLO_QuantizedIntTensor,
  HLO_ScalarIntTensor,
  HLO_StaticShapeTensor,
  HLO_Tensor,
  HLO_TensorOrToken,
  HLO_TensorOrTokenOrTuple,
  HLO_Token,
  HLO_Tuple,
  I32Tensor,
  Index,
  Shape_WitnessType,
  TensorOrMemref,
  anonymous_526,
  anonymous_610,
  anonymous_621,
  anonymous_653,
  anonymous_655,
  anonymous_673,
  anonymous_686,
  anonymous_688,
  anonymous_694,
  anonymous_704
};

class AttributeGeneratorBase {
public:
  AttributeGeneratorBase(mlir::MLIRContext &ctx) : ctx(ctx) {}
  virtual ~AttributeGeneratorBase() = default;

  // AttrDef generators. (exhaustively synthesize all combinations)
  std::vector<mlir::Attribute> genChloComparisonDirectionAttr(const std::string attributeName);
  std::vector<mlir::Attribute> genChloComparisonTypeAttr(const std::string attributeName);
  std::vector<mlir::Attribute> genLinalgBinaryFnAttr(const std::string attributeName);
  std::vector<mlir::Attribute> genLinalgTypeFnAttr(const std::string attributeName);
  std::vector<mlir::Attribute> genLinalgUnaryFnAttr(const std::string attributeName);
  std::vector<mlir::Attribute> genStablehloArgResultAliasAttr(const std::string attributeName);
  std::vector<mlir::Attribute> genStablehloChannelHandleAttr(const std::string attributeName);
  std::vector<mlir::Attribute> genStablehloComparisonDirectionAttr(const std::string attributeName);
  std::vector<mlir::Attribute> genStablehloComparisonTypeAttr(const std::string attributeName);
  std::vector<mlir::Attribute> genStablehloConvDimensionNumbersAttr(const std::string attributeName);
  std::vector<mlir::Attribute> genStablehloDotDimensionNumbersAttr(const std::string attributeName);
  std::vector<mlir::Attribute> genStablehloFftTypeAttr(const std::string attributeName);
  std::vector<mlir::Attribute> genStablehloGatherDimensionNumbersAttr(const std::string attributeName);
  std::vector<mlir::Attribute> genStablehloPrecisionAttr(const std::string attributeName);
  std::vector<mlir::Attribute> genStablehloRngAlgorithmAttr(const std::string attributeName);
  std::vector<mlir::Attribute> genStablehloRngDistributionAttr(const std::string attributeName);
  std::vector<mlir::Attribute> genStablehloScatterDimensionNumbersAttr(const std::string attributeName);
  std::vector<mlir::Attribute> genStablehloTransposeAttr(const std::string attributeName);
  std::vector<mlir::Attribute> genStablehloTypeExtensionsAttr(const std::string attributeName);

  // Attr generators. (to be derived and implemented)
  virtual std::vector<mlir::Attribute> genArrayAttr(const std::string attributeName);
  virtual std::vector<mlir::Attribute> genBinaryFnAttr(const std::string attributeName);
  virtual std::vector<mlir::Attribute> genBoolAttr(const std::string attributeName);
  virtual std::vector<mlir::Attribute> genChannelHandleAttr(const std::string attributeName);
  virtual std::vector<mlir::Attribute> genComparisonTypeAttr(const std::string attributeName);
  virtual std::vector<mlir::Attribute> genCustomCallApiVersionAttr(const std::string attributeName);
  virtual std::vector<mlir::Attribute> genDenseElementsAttr(const std::string attributeName);
  virtual std::vector<mlir::Attribute> genDenseI64ArrayAttr(const std::string attributeName);
  virtual std::vector<mlir::Attribute> genDenseIntElementsAttr(const std::string attributeName);
  virtual std::vector<mlir::Attribute> genElementsAttr(const std::string attributeName);
  virtual std::vector<mlir::Attribute> genFloatAttr(const std::string attributeName);
  virtual std::vector<mlir::Attribute> genIntegerAttr(const std::string attributeName);
  virtual std::vector<mlir::Attribute> genStringAttr(const std::string attributeName);
  virtual std::vector<mlir::Attribute> genTypeFnAttr(const std::string attributeName);
  virtual std::vector<mlir::Attribute> genTypedAttr(const std::string attributeName);
  virtual std::vector<mlir::Attribute> genUnaryFnAttr(const std::string attributeName);
  virtual std::vector<mlir::Attribute> genUnitAttr(const std::string attributeName);

  // Types used in enums. (to be derived and implemented)
  virtual std::vector<::llvm::SmallVector<int64_t>> genLlvmSmallVectorint64t(const std::string attributeName);
  virtual std::vector<bool> genBool(const std::string attributeName);
  virtual std::vector<int64_t> genInt64t(const std::string attributeName);

protected:
  mlir::MLIRContext &ctx;
};
using AttributeGeneratorBasePtr = std::shared_ptr<AttributeGeneratorBase>;

class GrammarOp {
public:
  virtual ~GrammarOp() {}
  virtual unsigned getNumOperands() const = 0;
  virtual unsigned getNumAttributes() const = 0;
  virtual unsigned getNumRegions() const = 0;
  virtual unsigned getNumResults() const = 0;
  virtual OpAndResType getOperandType(unsigned index) const = 0;
  virtual std::string getOperandName(unsigned index) const = 0;
  virtual mlir::Attribute getAttributeType(unsigned index) const = 0;
  virtual std::string getAttributeName(unsigned index) const = 0;
  virtual bool isAttributeRequired(unsigned index) const = 0;
  virtual std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const = 0;
  virtual OpAndResType getResultType(unsigned index) const = 0;
};
using GrammarOpPtr = std::unique_ptr<GrammarOp>;
std::string opAndResTypeToString(OpAndResType type);
GrammarOpPtr createGrammarOp(std::string name);

} // namespace grammar
#endif // IRSYNTH_GRAMMAR_H
