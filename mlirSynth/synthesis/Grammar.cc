/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Grammar (generated from tablegen)                                          *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/


#include "Grammar.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/TensorEncoding.h"
#include "stablehlo/dialect/Base.h"

// Include order matters
#include "stablehlo/dialect/ChloEnums.h.inc"
#define GET_ATTRDEF_CLASSES
#include "stablehlo/dialect/ChloAttrs.h.inc"

// Include order matters
#include "stablehlo/dialect/StablehloEnums.h.inc"
#define GET_ATTRDEF_CLASSES
#include "stablehlo/dialect/StablehloAttrs.h.inc"

#include "mlir/Dialect/Linalg/IR/Linalg.h"

#include <cassert>
#include <memory>
#include <string>

namespace grammar {
std::string opAndResTypeToString(OpAndResType type) {
  if (type == AnyRankedTensor) return "AnyRankedTensor";
  if (type == AnyShaped) return "AnyShaped";
  if (type == AnyTensor) return "AnyTensor";
  if (type == AnyType) return "AnyType";
  if (type == HLO_ComplexTensor) return "HLO_ComplexTensor";
  if (type == HLO_DimensionTensor) return "HLO_DimensionTensor";
  if (type == HLO_Fp32Or64Tensor) return "HLO_Fp32Or64Tensor";
  if (type == HLO_FpOrComplexTensor) return "HLO_FpOrComplexTensor";
  if (type == HLO_FpTensor) return "HLO_FpTensor";
  if (type == HLO_IntFpOrComplexTensor) return "HLO_IntFpOrComplexTensor";
  if (type == HLO_IntOrFpTensor) return "HLO_IntOrFpTensor";
  if (type == HLO_IntTensor) return "HLO_IntTensor";
  if (type == HLO_PredIntOrFpTensor) return "HLO_PredIntOrFpTensor";
  if (type == HLO_PredOrIntTensor) return "HLO_PredOrIntTensor";
  if (type == HLO_PredTensor) return "HLO_PredTensor";
  if (type == HLO_QuantizedIntTensor) return "HLO_QuantizedIntTensor";
  if (type == HLO_ScalarIntTensor) return "HLO_ScalarIntTensor";
  if (type == HLO_StaticShapeTensor) return "HLO_StaticShapeTensor";
  if (type == HLO_Tensor) return "HLO_Tensor";
  if (type == HLO_TensorOrToken) return "HLO_TensorOrToken";
  if (type == HLO_TensorOrTokenOrTuple) return "HLO_TensorOrTokenOrTuple";
  if (type == HLO_Token) return "HLO_Token";
  if (type == HLO_Tuple) return "HLO_Tuple";
  if (type == I32Tensor) return "I32Tensor";
  if (type == Index) return "Index";
  if (type == Shape_WitnessType) return "Shape_WitnessType";
  if (type == TensorOrMemref) return "TensorOrMemref";
  if (type == anonymous_526) return "anonymous_526";
  if (type == anonymous_610) return "anonymous_610";
  if (type == anonymous_621) return "anonymous_621";
  if (type == anonymous_653) return "anonymous_653";
  if (type == anonymous_655) return "anonymous_655";
  if (type == anonymous_673) return "anonymous_673";
  if (type == anonymous_686) return "anonymous_686";
  if (type == anonymous_688) return "anonymous_688";
  if (type == anonymous_694) return "anonymous_694";
  if (type == anonymous_704) return "anonymous_704";
  assert(false && "Invalid OpAndResType");
}

std::vector<mlir::Attribute> AttributeGeneratorBase::genLinalgBinaryFnAttr(const std::string attributeName) {
  std::vector<::mlir::linalg::BinaryFn> valueEnumerants = {
    ::mlir::linalg::BinaryFn::add,
    ::mlir::linalg::BinaryFn::sub,
    ::mlir::linalg::BinaryFn::mul,
    ::mlir::linalg::BinaryFn::max_signed,
    ::mlir::linalg::BinaryFn::min_signed,
    ::mlir::linalg::BinaryFn::max_unsigned,
    ::mlir::linalg::BinaryFn::min_unsigned,
  };
  std::vector<mlir::Attribute> ret;
  for (const auto &v0 : valueEnumerants) {
    ret.push_back(::mlir::linalg::BinaryFnAttr::get(&ctx, 
      v0));
  }
  return ret;
}

std::vector<mlir::Attribute> AttributeGeneratorBase::genChloComparisonDirectionAttr(const std::string attributeName) {
  std::vector<::mlir::chlo::ComparisonDirection> valueEnumerants = {
    ::mlir::chlo::ComparisonDirection::EQ,
    ::mlir::chlo::ComparisonDirection::NE,
    ::mlir::chlo::ComparisonDirection::GE,
    ::mlir::chlo::ComparisonDirection::GT,
    ::mlir::chlo::ComparisonDirection::LE,
    ::mlir::chlo::ComparisonDirection::LT,
  };
  std::vector<mlir::Attribute> ret;
  for (const auto &v0 : valueEnumerants) {
    ret.push_back(::mlir::chlo::ComparisonDirectionAttr::get(&ctx, 
      v0));
  }
  return ret;
}

std::vector<mlir::Attribute> AttributeGeneratorBase::genChloComparisonTypeAttr(const std::string attributeName) {
  std::vector<::mlir::chlo::ComparisonType> valueEnumerants = {
    ::mlir::chlo::ComparisonType::NOTYPE,
    ::mlir::chlo::ComparisonType::FLOAT,
    ::mlir::chlo::ComparisonType::TOTALORDER,
    ::mlir::chlo::ComparisonType::SIGNED,
    ::mlir::chlo::ComparisonType::UNSIGNED,
  };
  std::vector<mlir::Attribute> ret;
  for (const auto &v0 : valueEnumerants) {
    ret.push_back(::mlir::chlo::ComparisonTypeAttr::get(&ctx, 
      v0));
  }
  return ret;
}

std::vector<mlir::Attribute> AttributeGeneratorBase::genStablehloArgResultAliasAttr(const std::string attributeName) {
  std::vector<::llvm::SmallVector<int64_t>> argTupleIndicesEnumerants = genLlvmSmallVectorint64t(attributeName);
  std::vector<int64_t> resultIndexEnumerants = genInt64t(attributeName);
  std::vector<::llvm::SmallVector<int64_t>> resultTupleIndicesEnumerants = genLlvmSmallVectorint64t(attributeName);
  std::vector<bool> isMustAliasEnumerants = genBool(attributeName);
  std::vector<mlir::Attribute> ret;
  for (const auto &v0 : argTupleIndicesEnumerants) {
    for (const auto &v1 : resultIndexEnumerants) {
      for (const auto &v2 : resultTupleIndicesEnumerants) {
        for (const auto &v3 : isMustAliasEnumerants) {
          ret.push_back(::mlir::stablehlo::ArgResultAliasAttr::get(&ctx, 
            v0,
            v1,
            v2,
            v3));
        }
      }
    }
  }
  return ret;
}

std::vector<mlir::Attribute> AttributeGeneratorBase::genStablehloChannelHandleAttr(const std::string attributeName) {
  std::vector<int64_t> handleEnumerants = genInt64t(attributeName);
  std::vector<int64_t> typeEnumerants = genInt64t(attributeName);
  std::vector<mlir::Attribute> ret;
  for (const auto &v0 : handleEnumerants) {
    for (const auto &v1 : typeEnumerants) {
      ret.push_back(::mlir::stablehlo::ChannelHandleAttr::get(&ctx, 
        v0,
        v1));
    }
  }
  return ret;
}

std::vector<mlir::Attribute> AttributeGeneratorBase::genStablehloComparisonDirectionAttr(const std::string attributeName) {
  std::vector<::mlir::stablehlo::ComparisonDirection> valueEnumerants = {
    ::mlir::stablehlo::ComparisonDirection::EQ,
    ::mlir::stablehlo::ComparisonDirection::NE,
    ::mlir::stablehlo::ComparisonDirection::GE,
    ::mlir::stablehlo::ComparisonDirection::GT,
    ::mlir::stablehlo::ComparisonDirection::LE,
    ::mlir::stablehlo::ComparisonDirection::LT,
  };
  std::vector<mlir::Attribute> ret;
  for (const auto &v0 : valueEnumerants) {
    ret.push_back(::mlir::stablehlo::ComparisonDirectionAttr::get(&ctx, 
      v0));
  }
  return ret;
}

std::vector<mlir::Attribute> AttributeGeneratorBase::genStablehloComparisonTypeAttr(const std::string attributeName) {
  std::vector<::mlir::stablehlo::ComparisonType> valueEnumerants = {
    ::mlir::stablehlo::ComparisonType::NOTYPE,
    ::mlir::stablehlo::ComparisonType::FLOAT,
    ::mlir::stablehlo::ComparisonType::TOTALORDER,
    ::mlir::stablehlo::ComparisonType::SIGNED,
    ::mlir::stablehlo::ComparisonType::UNSIGNED,
  };
  std::vector<mlir::Attribute> ret;
  for (const auto &v0 : valueEnumerants) {
    ret.push_back(::mlir::stablehlo::ComparisonTypeAttr::get(&ctx, 
      v0));
  }
  return ret;
}

std::vector<mlir::Attribute> AttributeGeneratorBase::genStablehloConvDimensionNumbersAttr(const std::string attributeName) {
  std::vector<int64_t> inputBatchDimensionEnumerants = genInt64t(attributeName);
  std::vector<int64_t> inputFeatureDimensionEnumerants = genInt64t(attributeName);
  std::vector<::llvm::SmallVector<int64_t>> inputSpatialDimensionsEnumerants = genLlvmSmallVectorint64t(attributeName);
  std::vector<int64_t> kernelInputFeatureDimensionEnumerants = genInt64t(attributeName);
  std::vector<int64_t> kernelOutputFeatureDimensionEnumerants = genInt64t(attributeName);
  std::vector<::llvm::SmallVector<int64_t>> kernelSpatialDimensionsEnumerants = genLlvmSmallVectorint64t(attributeName);
  std::vector<int64_t> outputBatchDimensionEnumerants = genInt64t(attributeName);
  std::vector<int64_t> outputFeatureDimensionEnumerants = genInt64t(attributeName);
  std::vector<::llvm::SmallVector<int64_t>> outputSpatialDimensionsEnumerants = genLlvmSmallVectorint64t(attributeName);
  std::vector<mlir::Attribute> ret;
  for (const auto &v0 : inputBatchDimensionEnumerants) {
    for (const auto &v1 : inputFeatureDimensionEnumerants) {
      for (const auto &v2 : inputSpatialDimensionsEnumerants) {
        for (const auto &v3 : kernelInputFeatureDimensionEnumerants) {
          for (const auto &v4 : kernelOutputFeatureDimensionEnumerants) {
            for (const auto &v5 : kernelSpatialDimensionsEnumerants) {
              for (const auto &v6 : outputBatchDimensionEnumerants) {
                for (const auto &v7 : outputFeatureDimensionEnumerants) {
                  for (const auto &v8 : outputSpatialDimensionsEnumerants) {
                    ret.push_back(::mlir::stablehlo::ConvDimensionNumbersAttr::get(&ctx, 
                      v0,
                      v1,
                      v2,
                      v3,
                      v4,
                      v5,
                      v6,
                      v7,
                      v8));
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  return ret;
}

std::vector<mlir::Attribute> AttributeGeneratorBase::genStablehloDotDimensionNumbersAttr(const std::string attributeName) {
  std::vector<::llvm::SmallVector<int64_t>> lhsBatchingDimensionsEnumerants = genLlvmSmallVectorint64t(attributeName);
  std::vector<::llvm::SmallVector<int64_t>> rhsBatchingDimensionsEnumerants = genLlvmSmallVectorint64t(attributeName);
  std::vector<::llvm::SmallVector<int64_t>> lhsContractingDimensionsEnumerants = genLlvmSmallVectorint64t(attributeName);
  std::vector<::llvm::SmallVector<int64_t>> rhsContractingDimensionsEnumerants = genLlvmSmallVectorint64t(attributeName);
  std::vector<mlir::Attribute> ret;
  for (const auto &v0 : lhsBatchingDimensionsEnumerants) {
    for (const auto &v1 : rhsBatchingDimensionsEnumerants) {
      for (const auto &v2 : lhsContractingDimensionsEnumerants) {
        for (const auto &v3 : rhsContractingDimensionsEnumerants) {
          ret.push_back(::mlir::stablehlo::DotDimensionNumbersAttr::get(&ctx, 
            v0,
            v1,
            v2,
            v3));
        }
      }
    }
  }
  return ret;
}

std::vector<mlir::Attribute> AttributeGeneratorBase::genStablehloFftTypeAttr(const std::string attributeName) {
  std::vector<::mlir::stablehlo::FftType> valueEnumerants = {
    ::mlir::stablehlo::FftType::FFT,
    ::mlir::stablehlo::FftType::IFFT,
    ::mlir::stablehlo::FftType::RFFT,
    ::mlir::stablehlo::FftType::IRFFT,
  };
  std::vector<mlir::Attribute> ret;
  for (const auto &v0 : valueEnumerants) {
    ret.push_back(::mlir::stablehlo::FftTypeAttr::get(&ctx, 
      v0));
  }
  return ret;
}

std::vector<mlir::Attribute> AttributeGeneratorBase::genStablehloGatherDimensionNumbersAttr(const std::string attributeName) {
  std::vector<::llvm::SmallVector<int64_t>> offsetDimsEnumerants = genLlvmSmallVectorint64t(attributeName);
  std::vector<::llvm::SmallVector<int64_t>> collapsedSliceDimsEnumerants = genLlvmSmallVectorint64t(attributeName);
  std::vector<::llvm::SmallVector<int64_t>> startIndexMapEnumerants = genLlvmSmallVectorint64t(attributeName);
  std::vector<int64_t> indexVectorDimEnumerants = genInt64t(attributeName);
  std::vector<mlir::Attribute> ret;
  for (const auto &v0 : offsetDimsEnumerants) {
    for (const auto &v1 : collapsedSliceDimsEnumerants) {
      for (const auto &v2 : startIndexMapEnumerants) {
        for (const auto &v3 : indexVectorDimEnumerants) {
          ret.push_back(::mlir::stablehlo::GatherDimensionNumbersAttr::get(&ctx, 
            v0,
            v1,
            v2,
            v3));
        }
      }
    }
  }
  return ret;
}

std::vector<mlir::Attribute> AttributeGeneratorBase::genStablehloPrecisionAttr(const std::string attributeName) {
  std::vector<::mlir::stablehlo::Precision> valueEnumerants = {
    ::mlir::stablehlo::Precision::DEFAULT,
    ::mlir::stablehlo::Precision::HIGH,
    ::mlir::stablehlo::Precision::HIGHEST,
  };
  std::vector<mlir::Attribute> ret;
  for (const auto &v0 : valueEnumerants) {
    ret.push_back(::mlir::stablehlo::PrecisionAttr::get(&ctx, 
      v0));
  }
  return ret;
}

std::vector<mlir::Attribute> AttributeGeneratorBase::genStablehloRngAlgorithmAttr(const std::string attributeName) {
  std::vector<::mlir::stablehlo::RngAlgorithm> valueEnumerants = {
    ::mlir::stablehlo::RngAlgorithm::DEFAULT,
    ::mlir::stablehlo::RngAlgorithm::THREE_FRY,
    ::mlir::stablehlo::RngAlgorithm::PHILOX,
  };
  std::vector<mlir::Attribute> ret;
  for (const auto &v0 : valueEnumerants) {
    ret.push_back(::mlir::stablehlo::RngAlgorithmAttr::get(&ctx, 
      v0));
  }
  return ret;
}

std::vector<mlir::Attribute> AttributeGeneratorBase::genStablehloRngDistributionAttr(const std::string attributeName) {
  std::vector<::mlir::stablehlo::RngDistribution> valueEnumerants = {
    ::mlir::stablehlo::RngDistribution::UNIFORM,
    ::mlir::stablehlo::RngDistribution::NORMAL,
  };
  std::vector<mlir::Attribute> ret;
  for (const auto &v0 : valueEnumerants) {
    ret.push_back(::mlir::stablehlo::RngDistributionAttr::get(&ctx, 
      v0));
  }
  return ret;
}

std::vector<mlir::Attribute> AttributeGeneratorBase::genStablehloScatterDimensionNumbersAttr(const std::string attributeName) {
  std::vector<::llvm::SmallVector<int64_t>> updateWindowDimsEnumerants = genLlvmSmallVectorint64t(attributeName);
  std::vector<::llvm::SmallVector<int64_t>> insertedWindowDimsEnumerants = genLlvmSmallVectorint64t(attributeName);
  std::vector<::llvm::SmallVector<int64_t>> scatterDimsToOperandDimsEnumerants = genLlvmSmallVectorint64t(attributeName);
  std::vector<int64_t> indexVectorDimEnumerants = genInt64t(attributeName);
  std::vector<mlir::Attribute> ret;
  for (const auto &v0 : updateWindowDimsEnumerants) {
    for (const auto &v1 : insertedWindowDimsEnumerants) {
      for (const auto &v2 : scatterDimsToOperandDimsEnumerants) {
        for (const auto &v3 : indexVectorDimEnumerants) {
          ret.push_back(::mlir::stablehlo::ScatterDimensionNumbersAttr::get(&ctx, 
            v0,
            v1,
            v2,
            v3));
        }
      }
    }
  }
  return ret;
}

std::vector<mlir::Attribute> AttributeGeneratorBase::genStablehloTransposeAttr(const std::string attributeName) {
  std::vector<::mlir::stablehlo::Transpose> valueEnumerants = {
    ::mlir::stablehlo::Transpose::TRANSPOSE_INVALID,
    ::mlir::stablehlo::Transpose::NO_TRANSPOSE,
    ::mlir::stablehlo::Transpose::TRANSPOSE,
    ::mlir::stablehlo::Transpose::ADJOINT,
  };
  std::vector<mlir::Attribute> ret;
  for (const auto &v0 : valueEnumerants) {
    ret.push_back(::mlir::stablehlo::TransposeAttr::get(&ctx, 
      v0));
  }
  return ret;
}

std::vector<mlir::Attribute> AttributeGeneratorBase::genStablehloTypeExtensionsAttr(const std::string attributeName) {
  std::vector<::llvm::SmallVector<int64_t>> boundsEnumerants = genLlvmSmallVectorint64t(attributeName);
  std::vector<mlir::Attribute> ret;
  for (const auto &v0 : boundsEnumerants) {
    ret.push_back(::mlir::stablehlo::TypeExtensionsAttr::get(&ctx, 
      v0));
  }
  return ret;
}

std::vector<mlir::Attribute> AttributeGeneratorBase::genLinalgTypeFnAttr(const std::string attributeName) {
  std::vector<::mlir::linalg::TypeFn> valueEnumerants = {
    ::mlir::linalg::TypeFn::cast_signed,
    ::mlir::linalg::TypeFn::cast_unsigned,
  };
  std::vector<mlir::Attribute> ret;
  for (const auto &v0 : valueEnumerants) {
    ret.push_back(::mlir::linalg::TypeFnAttr::get(&ctx, 
      v0));
  }
  return ret;
}

std::vector<mlir::Attribute> AttributeGeneratorBase::genLinalgUnaryFnAttr(const std::string attributeName) {
  std::vector<::mlir::linalg::UnaryFn> valueEnumerants = {
    ::mlir::linalg::UnaryFn::exp,
    ::mlir::linalg::UnaryFn::log,
    ::mlir::linalg::UnaryFn::abs,
    ::mlir::linalg::UnaryFn::ceil,
    ::mlir::linalg::UnaryFn::floor,
    ::mlir::linalg::UnaryFn::negf,
  };
  std::vector<mlir::Attribute> ret;
  for (const auto &v0 : valueEnumerants) {
    ret.push_back(::mlir::linalg::UnaryFnAttr::get(&ctx, 
      v0));
  }
  return ret;
}

std::vector<mlir::Attribute> AttributeGeneratorBase::genArrayAttr(const std::string attributeName) {
  llvm::outs() << "WARNING: Not implemented: genArrayAttr\n";
  std::vector<mlir::Attribute> ret;
  return ret;
}

std::vector<mlir::Attribute> AttributeGeneratorBase::genBinaryFnAttr(const std::string attributeName) {
  llvm::outs() << "WARNING: Not implemented: genBinaryFnAttr\n";
  std::vector<mlir::Attribute> ret;
  return ret;
}

std::vector<mlir::Attribute> AttributeGeneratorBase::genBoolAttr(const std::string attributeName) {
  llvm::outs() << "WARNING: Not implemented: genBoolAttr\n";
  std::vector<mlir::Attribute> ret;
  return ret;
}

std::vector<mlir::Attribute> AttributeGeneratorBase::genChannelHandleAttr(const std::string attributeName) {
  llvm::outs() << "WARNING: Not implemented: genChannelHandleAttr\n";
  std::vector<mlir::Attribute> ret;
  return ret;
}

std::vector<mlir::Attribute> AttributeGeneratorBase::genComparisonTypeAttr(const std::string attributeName) {
  llvm::outs() << "WARNING: Not implemented: genComparisonTypeAttr\n";
  std::vector<mlir::Attribute> ret;
  return ret;
}

std::vector<mlir::Attribute> AttributeGeneratorBase::genCustomCallApiVersionAttr(const std::string attributeName) {
  llvm::outs() << "WARNING: Not implemented: genCustomCallApiVersionAttr\n";
  std::vector<mlir::Attribute> ret;
  return ret;
}

std::vector<mlir::Attribute> AttributeGeneratorBase::genDenseElementsAttr(const std::string attributeName) {
  llvm::outs() << "WARNING: Not implemented: genDenseElementsAttr\n";
  std::vector<mlir::Attribute> ret;
  return ret;
}

std::vector<mlir::Attribute> AttributeGeneratorBase::genDenseI64ArrayAttr(const std::string attributeName) {
  llvm::outs() << "WARNING: Not implemented: genDenseI64ArrayAttr\n";
  std::vector<mlir::Attribute> ret;
  return ret;
}

std::vector<mlir::Attribute> AttributeGeneratorBase::genDenseIntElementsAttr(const std::string attributeName) {
  llvm::outs() << "WARNING: Not implemented: genDenseIntElementsAttr\n";
  std::vector<mlir::Attribute> ret;
  return ret;
}

std::vector<mlir::Attribute> AttributeGeneratorBase::genElementsAttr(const std::string attributeName) {
  llvm::outs() << "WARNING: Not implemented: genElementsAttr\n";
  std::vector<mlir::Attribute> ret;
  return ret;
}

std::vector<mlir::Attribute> AttributeGeneratorBase::genFloatAttr(const std::string attributeName) {
  llvm::outs() << "WARNING: Not implemented: genFloatAttr\n";
  std::vector<mlir::Attribute> ret;
  return ret;
}

std::vector<mlir::Attribute> AttributeGeneratorBase::genIntegerAttr(const std::string attributeName) {
  llvm::outs() << "WARNING: Not implemented: genIntegerAttr\n";
  std::vector<mlir::Attribute> ret;
  return ret;
}

std::vector<mlir::Attribute> AttributeGeneratorBase::genStringAttr(const std::string attributeName) {
  llvm::outs() << "WARNING: Not implemented: genStringAttr\n";
  std::vector<mlir::Attribute> ret;
  return ret;
}

std::vector<mlir::Attribute> AttributeGeneratorBase::genTypeFnAttr(const std::string attributeName) {
  llvm::outs() << "WARNING: Not implemented: genTypeFnAttr\n";
  std::vector<mlir::Attribute> ret;
  return ret;
}

std::vector<mlir::Attribute> AttributeGeneratorBase::genTypedAttr(const std::string attributeName) {
  llvm::outs() << "WARNING: Not implemented: genTypedAttr\n";
  std::vector<mlir::Attribute> ret;
  return ret;
}

std::vector<mlir::Attribute> AttributeGeneratorBase::genUnaryFnAttr(const std::string attributeName) {
  llvm::outs() << "WARNING: Not implemented: genUnaryFnAttr\n";
  std::vector<mlir::Attribute> ret;
  return ret;
}

std::vector<mlir::Attribute> AttributeGeneratorBase::genUnitAttr(const std::string attributeName) {
  llvm::outs() << "WARNING: Not implemented: genUnitAttr\n";
  std::vector<mlir::Attribute> ret;
  return ret;
}

std::vector<::llvm::SmallVector<int64_t>> AttributeGeneratorBase::genLlvmSmallVectorint64t(const std::string attributeName) {
  llvm::outs() << "WARNING: Not implemented: genLlvmSmallVectorint64t\n";
  std::vector<::llvm::SmallVector<int64_t>> ret;
  return ret;
}

std::vector<bool> AttributeGeneratorBase::genBool(const std::string attributeName) {
  llvm::outs() << "WARNING: Not implemented: genBool\n";
  std::vector<bool> ret;
  return ret;
}

std::vector<int64_t> AttributeGeneratorBase::genInt64t(const std::string attributeName) {
  llvm::outs() << "WARNING: Not implemented: genInt64t\n";
  std::vector<int64_t> ret;
  return ret;
}

class linalg_batch_matmul : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 2; }
  unsigned getNumAttributes() const override { return 0; }
  unsigned getNumRegions() const override { return 1; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return AnyType;
      case 1: return AnyShaped;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "inputs";
      case 1: return "outputs";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return AnyRankedTensor;
    }
    assert(false && "Invalid result index");
  }
};

class linalg_batch_matvec : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 2; }
  unsigned getNumAttributes() const override { return 0; }
  unsigned getNumRegions() const override { return 1; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return AnyType;
      case 1: return AnyShaped;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "inputs";
      case 1: return "outputs";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return AnyRankedTensor;
    }
    assert(false && "Invalid result index");
  }
};

class linalg_batch_reduce_matmul : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 2; }
  unsigned getNumAttributes() const override { return 0; }
  unsigned getNumRegions() const override { return 1; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return AnyType;
      case 1: return AnyShaped;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "inputs";
      case 1: return "outputs";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return AnyRankedTensor;
    }
    assert(false && "Invalid result index");
  }
};

class chlo_acos : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 1; }
  unsigned getNumAttributes() const override { return 0; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpOrComplexTensor;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "operand";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpOrComplexTensor;
    }
    assert(false && "Invalid result index");
  }
};

class chlo_acosh : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 1; }
  unsigned getNumAttributes() const override { return 0; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpOrComplexTensor;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "operand";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpOrComplexTensor;
    }
    assert(false && "Invalid result index");
  }
};

class chlo_asin : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 1; }
  unsigned getNumAttributes() const override { return 0; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpOrComplexTensor;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "operand";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpOrComplexTensor;
    }
    assert(false && "Invalid result index");
  }
};

class chlo_asinh : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 1; }
  unsigned getNumAttributes() const override { return 0; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpOrComplexTensor;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "operand";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpOrComplexTensor;
    }
    assert(false && "Invalid result index");
  }
};

class chlo_atan : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 1; }
  unsigned getNumAttributes() const override { return 0; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpOrComplexTensor;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "operand";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpOrComplexTensor;
    }
    assert(false && "Invalid result index");
  }
};

class chlo_atanh : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 1; }
  unsigned getNumAttributes() const override { return 0; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpOrComplexTensor;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "operand";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpOrComplexTensor;
    }
    assert(false && "Invalid result index");
  }
};

class chlo_bessel_i1e : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 1; }
  unsigned getNumAttributes() const override { return 0; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpOrComplexTensor;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "operand";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpOrComplexTensor;
    }
    assert(false && "Invalid result index");
  }
};

class chlo_broadcast_add : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 2; }
  unsigned getNumAttributes() const override { return 1; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
      case 1: return HLO_Tensor;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "lhs";
      case 1: return "rhs";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return ::mlir::DenseIntElementsAttr();
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
      case 0: return "broadcast_dimensions";
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
      case 0: return false;
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    // attrs.push_back(attrGen->genDenseIntElementsAttr("broadcast_dimensions"));
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class chlo_broadcast_and : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 2; }
  unsigned getNumAttributes() const override { return 1; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_PredOrIntTensor;
      case 1: return HLO_PredOrIntTensor;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "lhs";
      case 1: return "rhs";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return ::mlir::DenseIntElementsAttr();
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
      case 0: return "broadcast_dimensions";
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
      case 0: return false;
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    // attrs.push_back(attrGen->genDenseIntElementsAttr("broadcast_dimensions"));
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class chlo_broadcast_atan2 : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 2; }
  unsigned getNumAttributes() const override { return 1; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
      case 1: return HLO_Tensor;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "lhs";
      case 1: return "rhs";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return ::mlir::DenseIntElementsAttr();
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
      case 0: return "broadcast_dimensions";
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
      case 0: return false;
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    // attrs.push_back(attrGen->genDenseIntElementsAttr("broadcast_dimensions"));
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class chlo_broadcast_compare : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 2; }
  unsigned getNumAttributes() const override { return 3; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
      case 1: return HLO_Tensor;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "lhs";
      case 1: return "rhs";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return ::mlir::DenseIntElementsAttr();
      case 1: return ::mlir::chlo::ComparisonDirectionAttr();
      case 2: return ::mlir::chlo::ComparisonTypeAttr();
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
      case 0: return "broadcast_dimensions";
      case 1: return "comparison_direction";
      case 2: return "compare_type";
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
      case 0: return false;
      case 1: return true;
      case 2: return false;
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    // attrs.push_back(attrGen->genDenseIntElementsAttr("broadcast_dimensions"));
    attrs.push_back(attrGen->genChloComparisonDirectionAttr("comparison_direction"));
    // attrs.push_back(attrGen->genComparisonTypeAttr("compare_type"));
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_PredTensor;
    }
    assert(false && "Invalid result index");
  }
};

class chlo_broadcast_complex : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 2; }
  unsigned getNumAttributes() const override { return 1; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpTensor;
      case 1: return HLO_FpTensor;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "lhs";
      case 1: return "rhs";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return ::mlir::DenseIntElementsAttr();
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
      case 0: return "broadcast_dimensions";
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
      case 0: return false;
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    // attrs.push_back(attrGen->genDenseIntElementsAttr("broadcast_dimensions"));
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_ComplexTensor;
    }
    assert(false && "Invalid result index");
  }
};

class chlo_broadcast_divide : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 2; }
  unsigned getNumAttributes() const override { return 1; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
      case 1: return HLO_Tensor;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "lhs";
      case 1: return "rhs";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return ::mlir::DenseIntElementsAttr();
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
      case 0: return "broadcast_dimensions";
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
      case 0: return false;
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    // attrs.push_back(attrGen->genDenseIntElementsAttr("broadcast_dimensions"));
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class chlo_broadcast_maximum : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 2; }
  unsigned getNumAttributes() const override { return 1; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
      case 1: return HLO_Tensor;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "lhs";
      case 1: return "rhs";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return ::mlir::DenseIntElementsAttr();
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
      case 0: return "broadcast_dimensions";
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
      case 0: return false;
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    // attrs.push_back(attrGen->genDenseIntElementsAttr("broadcast_dimensions"));
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class chlo_broadcast_minimum : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 2; }
  unsigned getNumAttributes() const override { return 1; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
      case 1: return HLO_Tensor;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "lhs";
      case 1: return "rhs";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return ::mlir::DenseIntElementsAttr();
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
      case 0: return "broadcast_dimensions";
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
      case 0: return false;
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    // attrs.push_back(attrGen->genDenseIntElementsAttr("broadcast_dimensions"));
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class chlo_broadcast_multiply : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 2; }
  unsigned getNumAttributes() const override { return 1; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
      case 1: return HLO_Tensor;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "lhs";
      case 1: return "rhs";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return ::mlir::DenseIntElementsAttr();
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
      case 0: return "broadcast_dimensions";
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
      case 0: return false;
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    // attrs.push_back(attrGen->genDenseIntElementsAttr("broadcast_dimensions"));
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class chlo_broadcast_next_after : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 2; }
  unsigned getNumAttributes() const override { return 1; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
      case 1: return HLO_Tensor;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "lhs";
      case 1: return "rhs";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return ::mlir::DenseIntElementsAttr();
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
      case 0: return "broadcast_dimensions";
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
      case 0: return false;
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    // attrs.push_back(attrGen->genDenseIntElementsAttr("broadcast_dimensions"));
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class chlo_broadcast_or : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 2; }
  unsigned getNumAttributes() const override { return 1; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_PredOrIntTensor;
      case 1: return HLO_PredOrIntTensor;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "lhs";
      case 1: return "rhs";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return ::mlir::DenseIntElementsAttr();
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
      case 0: return "broadcast_dimensions";
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
      case 0: return false;
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    // attrs.push_back(attrGen->genDenseIntElementsAttr("broadcast_dimensions"));
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class chlo_broadcast_polygamma : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 2; }
  unsigned getNumAttributes() const override { return 1; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
      case 1: return HLO_Tensor;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "lhs";
      case 1: return "rhs";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return ::mlir::DenseIntElementsAttr();
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
      case 0: return "broadcast_dimensions";
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
      case 0: return false;
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    // attrs.push_back(attrGen->genDenseIntElementsAttr("broadcast_dimensions"));
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class chlo_broadcast_power : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 2; }
  unsigned getNumAttributes() const override { return 1; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
      case 1: return HLO_Tensor;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "lhs";
      case 1: return "rhs";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return ::mlir::DenseIntElementsAttr();
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
      case 0: return "broadcast_dimensions";
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
      case 0: return false;
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    // attrs.push_back(attrGen->genDenseIntElementsAttr("broadcast_dimensions"));
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class chlo_broadcast_remainder : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 2; }
  unsigned getNumAttributes() const override { return 1; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
      case 1: return HLO_Tensor;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "lhs";
      case 1: return "rhs";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return ::mlir::DenseIntElementsAttr();
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
      case 0: return "broadcast_dimensions";
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
      case 0: return false;
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    // attrs.push_back(attrGen->genDenseIntElementsAttr("broadcast_dimensions"));
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class chlo_broadcast_select : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 3; }
  unsigned getNumAttributes() const override { return 0; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_PredTensor;
      case 1: return HLO_Tensor;
      case 2: return HLO_Tensor;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "pred";
      case 1: return "on_true";
      case 2: return "on_false";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class chlo_broadcast_shift_left : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 2; }
  unsigned getNumAttributes() const override { return 1; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
      case 1: return HLO_Tensor;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "lhs";
      case 1: return "rhs";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return ::mlir::DenseIntElementsAttr();
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
      case 0: return "broadcast_dimensions";
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
      case 0: return false;
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    // attrs.push_back(attrGen->genDenseIntElementsAttr("broadcast_dimensions"));
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class chlo_broadcast_shift_right_arithmetic : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 2; }
  unsigned getNumAttributes() const override { return 1; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
      case 1: return HLO_Tensor;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "lhs";
      case 1: return "rhs";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return ::mlir::DenseIntElementsAttr();
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
      case 0: return "broadcast_dimensions";
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
      case 0: return false;
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    // attrs.push_back(attrGen->genDenseIntElementsAttr("broadcast_dimensions"));
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class chlo_broadcast_shift_right_logical : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 2; }
  unsigned getNumAttributes() const override { return 1; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
      case 1: return HLO_Tensor;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "lhs";
      case 1: return "rhs";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return ::mlir::DenseIntElementsAttr();
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
      case 0: return "broadcast_dimensions";
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
      case 0: return false;
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    // attrs.push_back(attrGen->genDenseIntElementsAttr("broadcast_dimensions"));
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class chlo_broadcast_subtract : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 2; }
  unsigned getNumAttributes() const override { return 1; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
      case 1: return HLO_Tensor;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "lhs";
      case 1: return "rhs";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return ::mlir::DenseIntElementsAttr();
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
      case 0: return "broadcast_dimensions";
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
      case 0: return false;
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    // attrs.push_back(attrGen->genDenseIntElementsAttr("broadcast_dimensions"));
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class chlo_broadcast_xor : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 2; }
  unsigned getNumAttributes() const override { return 1; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_PredOrIntTensor;
      case 1: return HLO_PredOrIntTensor;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "lhs";
      case 1: return "rhs";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return ::mlir::DenseIntElementsAttr();
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
      case 0: return "broadcast_dimensions";
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
      case 0: return false;
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    // attrs.push_back(attrGen->genDenseIntElementsAttr("broadcast_dimensions"));
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class chlo_broadcast_zeta : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 2; }
  unsigned getNumAttributes() const override { return 1; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpTensor;
      case 1: return HLO_FpTensor;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "lhs";
      case 1: return "rhs";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return ::mlir::DenseIntElementsAttr();
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
      case 0: return "broadcast_dimensions";
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
      case 0: return false;
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    // attrs.push_back(attrGen->genDenseIntElementsAttr("broadcast_dimensions"));
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpTensor;
    }
    assert(false && "Invalid result index");
  }
};

class chlo_conj : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 1; }
  unsigned getNumAttributes() const override { return 0; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpOrComplexTensor;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "operand";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpOrComplexTensor;
    }
    assert(false && "Invalid result index");
  }
};

class chlo_constant_like : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 1; }
  unsigned getNumAttributes() const override { return 1; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "operand";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return ::mlir::TypedAttr();
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
      case 0: return "value";
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
      case 0: return true;
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    attrs.push_back(attrGen->genTypedAttr("value"));
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class chlo_constant : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 0; }
  unsigned getNumAttributes() const override { return 1; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return ::mlir::ElementsAttr();
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
      case 0: return "value";
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
      case 0: return true;
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    attrs.push_back(attrGen->genElementsAttr("value"));
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_StaticShapeTensor;
    }
    assert(false && "Invalid result index");
  }
};

class chlo_cosh : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 1; }
  unsigned getNumAttributes() const override { return 0; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpOrComplexTensor;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "operand";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpOrComplexTensor;
    }
    assert(false && "Invalid result index");
  }
};

class chlo_digamma : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 1; }
  unsigned getNumAttributes() const override { return 0; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpTensor;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "operand";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpTensor;
    }
    assert(false && "Invalid result index");
  }
};

class chlo_dynamic_reshape : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 2; }
  unsigned getNumAttributes() const override { return 0; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
      case 1: return HLO_DimensionTensor;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "operand";
      case 1: return "output_shape";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class chlo_erf : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 1; }
  unsigned getNumAttributes() const override { return 0; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpTensor;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "operand";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpTensor;
    }
    assert(false && "Invalid result index");
  }
};

class chlo_erfc : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 1; }
  unsigned getNumAttributes() const override { return 0; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpTensor;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "operand";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpTensor;
    }
    assert(false && "Invalid result index");
  }
};

class chlo_is_inf : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 1; }
  unsigned getNumAttributes() const override { return 0; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpTensor;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "operand";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_PredTensor;
    }
    assert(false && "Invalid result index");
  }
};

class chlo_is_neg_inf : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 1; }
  unsigned getNumAttributes() const override { return 0; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpTensor;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "operand";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_PredTensor;
    }
    assert(false && "Invalid result index");
  }
};

class chlo_is_pos_inf : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 1; }
  unsigned getNumAttributes() const override { return 0; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpTensor;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "operand";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_PredTensor;
    }
    assert(false && "Invalid result index");
  }
};

class chlo_lgamma : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 1; }
  unsigned getNumAttributes() const override { return 0; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpTensor;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "operand";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpTensor;
    }
    assert(false && "Invalid result index");
  }
};

class chlo_minimum_broadcast_shapes : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 1; }
  unsigned getNumAttributes() const override { return 0; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return anonymous_526;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "shapes";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return anonymous_526;
    }
    assert(false && "Invalid result index");
  }
};

class chlo_next_after : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 2; }
  unsigned getNumAttributes() const override { return 0; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpTensor;
      case 1: return HLO_FpTensor;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "x";
      case 1: return "y";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpTensor;
    }
    assert(false && "Invalid result index");
  }
};

class chlo_polygamma : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 2; }
  unsigned getNumAttributes() const override { return 0; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpTensor;
      case 1: return HLO_FpTensor;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "n";
      case 1: return "x";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpTensor;
    }
    assert(false && "Invalid result index");
  }
};

class chlo_rank_specialization_cluster : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 1; }
  unsigned getNumAttributes() const override { return 0; }
  unsigned getNumRegions() const override { return 1; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "operands";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class chlo_rank_specialization_cluster_yield : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 1; }
  unsigned getNumAttributes() const override { return 0; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 0; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "results";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid result index");
  }
};

class chlo_sinh : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 1; }
  unsigned getNumAttributes() const override { return 0; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpOrComplexTensor;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "operand";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpOrComplexTensor;
    }
    assert(false && "Invalid result index");
  }
};

class chlo_tan : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 1; }
  unsigned getNumAttributes() const override { return 0; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpOrComplexTensor;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "operand";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpOrComplexTensor;
    }
    assert(false && "Invalid result index");
  }
};

class chlo_top_k : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 1; }
  unsigned getNumAttributes() const override { return 1; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 2; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "operand";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return ::mlir::IntegerAttr();
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
      case 0: return "k";
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
      case 0: return true;
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    attrs.push_back(attrGen->genIntegerAttr("k"));
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
      case 1: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class chlo_zeta : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 2; }
  unsigned getNumAttributes() const override { return 0; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpTensor;
      case 1: return HLO_FpTensor;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "x";
      case 1: return "q";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpTensor;
    }
    assert(false && "Invalid result index");
  }
};

class linalg_conv_1d_ncw_fcw : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 2; }
  unsigned getNumAttributes() const override { return 2; }
  unsigned getNumRegions() const override { return 1; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return AnyType;
      case 1: return AnyShaped;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "inputs";
      case 1: return "outputs";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return ::mlir::DenseIntElementsAttr();
      case 1: return ::mlir::DenseIntElementsAttr();
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
      case 0: return "strides";
      case 1: return "dilations";
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
      case 0: return false;
      case 1: return false;
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    // attrs.push_back(attrGen->genDenseIntElementsAttr("strides"));
    // attrs.push_back(attrGen->genDenseIntElementsAttr("dilations"));
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return AnyRankedTensor;
    }
    assert(false && "Invalid result index");
  }
};

class linalg_conv_1d_nwc_wcf : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 2; }
  unsigned getNumAttributes() const override { return 2; }
  unsigned getNumRegions() const override { return 1; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return AnyType;
      case 1: return AnyShaped;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "inputs";
      case 1: return "outputs";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return ::mlir::DenseIntElementsAttr();
      case 1: return ::mlir::DenseIntElementsAttr();
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
      case 0: return "strides";
      case 1: return "dilations";
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
      case 0: return false;
      case 1: return false;
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    // attrs.push_back(attrGen->genDenseIntElementsAttr("strides"));
    // attrs.push_back(attrGen->genDenseIntElementsAttr("dilations"));
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return AnyRankedTensor;
    }
    assert(false && "Invalid result index");
  }
};

class linalg_conv_1d : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 2; }
  unsigned getNumAttributes() const override { return 0; }
  unsigned getNumRegions() const override { return 1; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return AnyType;
      case 1: return AnyShaped;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "inputs";
      case 1: return "outputs";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return AnyRankedTensor;
    }
    assert(false && "Invalid result index");
  }
};

class linalg_conv_2d_nchw_fchw : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 2; }
  unsigned getNumAttributes() const override { return 2; }
  unsigned getNumRegions() const override { return 1; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return AnyType;
      case 1: return AnyShaped;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "inputs";
      case 1: return "outputs";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return ::mlir::DenseIntElementsAttr();
      case 1: return ::mlir::DenseIntElementsAttr();
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
      case 0: return "strides";
      case 1: return "dilations";
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
      case 0: return false;
      case 1: return false;
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    // attrs.push_back(attrGen->genDenseIntElementsAttr("strides"));
    // attrs.push_back(attrGen->genDenseIntElementsAttr("dilations"));
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return AnyRankedTensor;
    }
    assert(false && "Invalid result index");
  }
};

class linalg_conv_2d_ngchw_fgchw : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 2; }
  unsigned getNumAttributes() const override { return 2; }
  unsigned getNumRegions() const override { return 1; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return AnyType;
      case 1: return AnyShaped;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "inputs";
      case 1: return "outputs";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return ::mlir::DenseIntElementsAttr();
      case 1: return ::mlir::DenseIntElementsAttr();
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
      case 0: return "strides";
      case 1: return "dilations";
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
      case 0: return false;
      case 1: return false;
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    // attrs.push_back(attrGen->genDenseIntElementsAttr("strides"));
    // attrs.push_back(attrGen->genDenseIntElementsAttr("dilations"));
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return AnyRankedTensor;
    }
    assert(false && "Invalid result index");
  }
};

class linalg_conv_2d_nhwc_fhwc : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 2; }
  unsigned getNumAttributes() const override { return 2; }
  unsigned getNumRegions() const override { return 1; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return AnyType;
      case 1: return AnyShaped;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "inputs";
      case 1: return "outputs";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return ::mlir::DenseIntElementsAttr();
      case 1: return ::mlir::DenseIntElementsAttr();
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
      case 0: return "strides";
      case 1: return "dilations";
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
      case 0: return false;
      case 1: return false;
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    // attrs.push_back(attrGen->genDenseIntElementsAttr("strides"));
    // attrs.push_back(attrGen->genDenseIntElementsAttr("dilations"));
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return AnyRankedTensor;
    }
    assert(false && "Invalid result index");
  }
};

class linalg_conv_2d_nhwc_hwcf : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 2; }
  unsigned getNumAttributes() const override { return 2; }
  unsigned getNumRegions() const override { return 1; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return AnyType;
      case 1: return AnyShaped;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "inputs";
      case 1: return "outputs";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return ::mlir::DenseIntElementsAttr();
      case 1: return ::mlir::DenseIntElementsAttr();
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
      case 0: return "strides";
      case 1: return "dilations";
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
      case 0: return false;
      case 1: return false;
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    // attrs.push_back(attrGen->genDenseIntElementsAttr("strides"));
    // attrs.push_back(attrGen->genDenseIntElementsAttr("dilations"));
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return AnyRankedTensor;
    }
    assert(false && "Invalid result index");
  }
};

class linalg_conv_2d_nhwc_hwcf_q : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 2; }
  unsigned getNumAttributes() const override { return 2; }
  unsigned getNumRegions() const override { return 1; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return AnyType;
      case 1: return AnyShaped;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "inputs";
      case 1: return "outputs";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return ::mlir::DenseIntElementsAttr();
      case 1: return ::mlir::DenseIntElementsAttr();
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
      case 0: return "strides";
      case 1: return "dilations";
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
      case 0: return false;
      case 1: return false;
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    // attrs.push_back(attrGen->genDenseIntElementsAttr("strides"));
    // attrs.push_back(attrGen->genDenseIntElementsAttr("dilations"));
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return AnyRankedTensor;
    }
    assert(false && "Invalid result index");
  }
};

class linalg_conv_2d : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 2; }
  unsigned getNumAttributes() const override { return 0; }
  unsigned getNumRegions() const override { return 1; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return AnyType;
      case 1: return AnyShaped;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "inputs";
      case 1: return "outputs";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return AnyRankedTensor;
    }
    assert(false && "Invalid result index");
  }
};

class linalg_conv_3d_ndhwc_dhwcf : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 2; }
  unsigned getNumAttributes() const override { return 2; }
  unsigned getNumRegions() const override { return 1; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return AnyType;
      case 1: return AnyShaped;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "inputs";
      case 1: return "outputs";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return ::mlir::DenseIntElementsAttr();
      case 1: return ::mlir::DenseIntElementsAttr();
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
      case 0: return "strides";
      case 1: return "dilations";
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
      case 0: return false;
      case 1: return false;
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    // attrs.push_back(attrGen->genDenseIntElementsAttr("strides"));
    // attrs.push_back(attrGen->genDenseIntElementsAttr("dilations"));
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return AnyRankedTensor;
    }
    assert(false && "Invalid result index");
  }
};

class linalg_conv_3d : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 2; }
  unsigned getNumAttributes() const override { return 0; }
  unsigned getNumRegions() const override { return 1; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return AnyType;
      case 1: return AnyShaped;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "inputs";
      case 1: return "outputs";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return AnyRankedTensor;
    }
    assert(false && "Invalid result index");
  }
};

class linalg_copy : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 2; }
  unsigned getNumAttributes() const override { return 1; }
  unsigned getNumRegions() const override { return 1; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return AnyType;
      case 1: return AnyShaped;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "inputs";
      case 1: return "outputs";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return ::mlir::linalg::TypeFnAttr();
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
      case 0: return "cast";
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
      case 0: return false;
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    // attrs.push_back(attrGen->genTypeFnAttr("cast"));
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return AnyRankedTensor;
    }
    assert(false && "Invalid result index");
  }
};

class linalg_depthwise_conv_1d_nwc_wc : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 2; }
  unsigned getNumAttributes() const override { return 2; }
  unsigned getNumRegions() const override { return 1; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return AnyType;
      case 1: return AnyShaped;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "inputs";
      case 1: return "outputs";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return ::mlir::DenseIntElementsAttr();
      case 1: return ::mlir::DenseIntElementsAttr();
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
      case 0: return "strides";
      case 1: return "dilations";
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
      case 0: return false;
      case 1: return false;
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    // attrs.push_back(attrGen->genDenseIntElementsAttr("strides"));
    // attrs.push_back(attrGen->genDenseIntElementsAttr("dilations"));
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return AnyRankedTensor;
    }
    assert(false && "Invalid result index");
  }
};

class linalg_depthwise_conv_1d_nwc_wcm : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 2; }
  unsigned getNumAttributes() const override { return 2; }
  unsigned getNumRegions() const override { return 1; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return AnyType;
      case 1: return AnyShaped;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "inputs";
      case 1: return "outputs";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return ::mlir::DenseIntElementsAttr();
      case 1: return ::mlir::DenseIntElementsAttr();
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
      case 0: return "strides";
      case 1: return "dilations";
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
      case 0: return false;
      case 1: return false;
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    // attrs.push_back(attrGen->genDenseIntElementsAttr("strides"));
    // attrs.push_back(attrGen->genDenseIntElementsAttr("dilations"));
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return AnyRankedTensor;
    }
    assert(false && "Invalid result index");
  }
};

class linalg_depthwise_conv_2d_nchw_chw : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 2; }
  unsigned getNumAttributes() const override { return 2; }
  unsigned getNumRegions() const override { return 1; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return AnyType;
      case 1: return AnyShaped;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "inputs";
      case 1: return "outputs";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return ::mlir::DenseIntElementsAttr();
      case 1: return ::mlir::DenseIntElementsAttr();
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
      case 0: return "strides";
      case 1: return "dilations";
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
      case 0: return false;
      case 1: return false;
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    // attrs.push_back(attrGen->genDenseIntElementsAttr("strides"));
    // attrs.push_back(attrGen->genDenseIntElementsAttr("dilations"));
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return AnyRankedTensor;
    }
    assert(false && "Invalid result index");
  }
};

class linalg_depthwise_conv_2d_nhwc_hwc : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 2; }
  unsigned getNumAttributes() const override { return 2; }
  unsigned getNumRegions() const override { return 1; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return AnyType;
      case 1: return AnyShaped;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "inputs";
      case 1: return "outputs";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return ::mlir::DenseIntElementsAttr();
      case 1: return ::mlir::DenseIntElementsAttr();
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
      case 0: return "strides";
      case 1: return "dilations";
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
      case 0: return false;
      case 1: return false;
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    // attrs.push_back(attrGen->genDenseIntElementsAttr("strides"));
    // attrs.push_back(attrGen->genDenseIntElementsAttr("dilations"));
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return AnyRankedTensor;
    }
    assert(false && "Invalid result index");
  }
};

class linalg_depthwise_conv_2d_nhwc_hwc_q : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 2; }
  unsigned getNumAttributes() const override { return 2; }
  unsigned getNumRegions() const override { return 1; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return AnyType;
      case 1: return AnyShaped;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "inputs";
      case 1: return "outputs";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return ::mlir::DenseIntElementsAttr();
      case 1: return ::mlir::DenseIntElementsAttr();
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
      case 0: return "strides";
      case 1: return "dilations";
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
      case 0: return false;
      case 1: return false;
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    // attrs.push_back(attrGen->genDenseIntElementsAttr("strides"));
    // attrs.push_back(attrGen->genDenseIntElementsAttr("dilations"));
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return AnyRankedTensor;
    }
    assert(false && "Invalid result index");
  }
};

class linalg_depthwise_conv_2d_nhwc_hwcm : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 2; }
  unsigned getNumAttributes() const override { return 2; }
  unsigned getNumRegions() const override { return 1; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return AnyType;
      case 1: return AnyShaped;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "inputs";
      case 1: return "outputs";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return ::mlir::DenseIntElementsAttr();
      case 1: return ::mlir::DenseIntElementsAttr();
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
      case 0: return "strides";
      case 1: return "dilations";
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
      case 0: return false;
      case 1: return false;
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    // attrs.push_back(attrGen->genDenseIntElementsAttr("strides"));
    // attrs.push_back(attrGen->genDenseIntElementsAttr("dilations"));
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return AnyRankedTensor;
    }
    assert(false && "Invalid result index");
  }
};

class linalg_depthwise_conv_2d_nhwc_hwcm_q : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 2; }
  unsigned getNumAttributes() const override { return 2; }
  unsigned getNumRegions() const override { return 1; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return AnyType;
      case 1: return AnyShaped;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "inputs";
      case 1: return "outputs";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return ::mlir::DenseIntElementsAttr();
      case 1: return ::mlir::DenseIntElementsAttr();
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
      case 0: return "strides";
      case 1: return "dilations";
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
      case 0: return false;
      case 1: return false;
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    // attrs.push_back(attrGen->genDenseIntElementsAttr("strides"));
    // attrs.push_back(attrGen->genDenseIntElementsAttr("dilations"));
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return AnyRankedTensor;
    }
    assert(false && "Invalid result index");
  }
};

class linalg_depthwise_conv_3d_ndhwc_dhwc : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 2; }
  unsigned getNumAttributes() const override { return 2; }
  unsigned getNumRegions() const override { return 1; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return AnyType;
      case 1: return AnyShaped;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "inputs";
      case 1: return "outputs";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return ::mlir::DenseIntElementsAttr();
      case 1: return ::mlir::DenseIntElementsAttr();
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
      case 0: return "strides";
      case 1: return "dilations";
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
      case 0: return false;
      case 1: return false;
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    // attrs.push_back(attrGen->genDenseIntElementsAttr("strides"));
    // attrs.push_back(attrGen->genDenseIntElementsAttr("dilations"));
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return AnyRankedTensor;
    }
    assert(false && "Invalid result index");
  }
};

class linalg_depthwise_conv_3d_ndhwc_dhwcm : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 2; }
  unsigned getNumAttributes() const override { return 2; }
  unsigned getNumRegions() const override { return 1; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return AnyType;
      case 1: return AnyShaped;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "inputs";
      case 1: return "outputs";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return ::mlir::DenseIntElementsAttr();
      case 1: return ::mlir::DenseIntElementsAttr();
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
      case 0: return "strides";
      case 1: return "dilations";
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
      case 0: return false;
      case 1: return false;
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    // attrs.push_back(attrGen->genDenseIntElementsAttr("strides"));
    // attrs.push_back(attrGen->genDenseIntElementsAttr("dilations"));
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return AnyRankedTensor;
    }
    assert(false && "Invalid result index");
  }
};

class linalg_dot : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 2; }
  unsigned getNumAttributes() const override { return 0; }
  unsigned getNumRegions() const override { return 1; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return AnyType;
      case 1: return AnyShaped;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "inputs";
      case 1: return "outputs";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return AnyRankedTensor;
    }
    assert(false && "Invalid result index");
  }
};

class linalg_elemwise_binary : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 2; }
  unsigned getNumAttributes() const override { return 2; }
  unsigned getNumRegions() const override { return 1; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return AnyType;
      case 1: return AnyShaped;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "inputs";
      case 1: return "outputs";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return ::mlir::linalg::BinaryFnAttr();
      case 1: return ::mlir::linalg::TypeFnAttr();
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
      case 0: return "fun";
      case 1: return "cast";
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
      case 0: return false;
      case 1: return false;
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    // attrs.push_back(attrGen->genBinaryFnAttr("fun"));
    // attrs.push_back(attrGen->genTypeFnAttr("cast"));
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return AnyRankedTensor;
    }
    assert(false && "Invalid result index");
  }
};

class linalg_elemwise_unary : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 2; }
  unsigned getNumAttributes() const override { return 2; }
  unsigned getNumRegions() const override { return 1; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return AnyType;
      case 1: return AnyShaped;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "inputs";
      case 1: return "outputs";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return ::mlir::linalg::UnaryFnAttr();
      case 1: return ::mlir::linalg::TypeFnAttr();
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
      case 0: return "fun";
      case 1: return "cast";
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
      case 0: return false;
      case 1: return false;
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    // attrs.push_back(attrGen->genUnaryFnAttr("fun"));
    // attrs.push_back(attrGen->genTypeFnAttr("cast"));
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return AnyRankedTensor;
    }
    assert(false && "Invalid result index");
  }
};

class linalg_fill : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 2; }
  unsigned getNumAttributes() const override { return 0; }
  unsigned getNumRegions() const override { return 1; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return AnyType;
      case 1: return AnyShaped;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "inputs";
      case 1: return "outputs";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return AnyRankedTensor;
    }
    assert(false && "Invalid result index");
  }
};

class linalg_fill_rng_2d : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 2; }
  unsigned getNumAttributes() const override { return 0; }
  unsigned getNumRegions() const override { return 1; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return AnyType;
      case 1: return AnyShaped;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "inputs";
      case 1: return "outputs";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return AnyRankedTensor;
    }
    assert(false && "Invalid result index");
  }
};

class linalg_generic : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 2; }
  unsigned getNumAttributes() const override { return 4; }
  unsigned getNumRegions() const override { return 1; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return AnyType;
      case 1: return AnyShaped;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "inputs";
      case 1: return "outputs";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return ::mlir::ArrayAttr();
      case 1: return ::mlir::ArrayAttr();
      case 2: return ::mlir::StringAttr();
      case 3: return ::mlir::StringAttr();
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
      case 0: return "indexing_maps";
      case 1: return "iterator_types";
      case 2: return "doc";
      case 3: return "library_call";
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
      case 0: return true;
      case 1: return true;
      case 2: return false;
      case 3: return false;
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    attrs.push_back(attrGen->genArrayAttr("indexing_maps"));
    attrs.push_back(attrGen->genArrayAttr("iterator_types"));
    // attrs.push_back(attrGen->genStringAttr("doc"));
    // attrs.push_back(attrGen->genStringAttr("library_call"));
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return AnyRankedTensor;
    }
    assert(false && "Invalid result index");
  }
};

class linalg_index : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 0; }
  unsigned getNumAttributes() const override { return 1; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return ::mlir::IntegerAttr();
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
      case 0: return "dim";
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
      case 0: return true;
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    attrs.push_back(attrGen->genIntegerAttr("dim"));
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return Index;
    }
    assert(false && "Invalid result index");
  }
};

class linalg_yield : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 1; }
  unsigned getNumAttributes() const override { return 0; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 0; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return AnyType;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "values";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid result index");
  }
};

class linalg_map : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 2; }
  unsigned getNumAttributes() const override { return 0; }
  unsigned getNumRegions() const override { return 1; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return TensorOrMemref;
      case 1: return TensorOrMemref;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "inputs";
      case 1: return "init";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return AnyTensor;
    }
    assert(false && "Invalid result index");
  }
};

class linalg_matmul : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 2; }
  unsigned getNumAttributes() const override { return 1; }
  unsigned getNumRegions() const override { return 1; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return AnyType;
      case 1: return AnyShaped;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "inputs";
      case 1: return "outputs";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return ::mlir::linalg::TypeFnAttr();
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
      case 0: return "cast";
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
      case 0: return false;
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    // attrs.push_back(attrGen->genTypeFnAttr("cast"));
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return AnyRankedTensor;
    }
    assert(false && "Invalid result index");
  }
};

class linalg_matmul_unsigned : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 2; }
  unsigned getNumAttributes() const override { return 0; }
  unsigned getNumRegions() const override { return 1; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return AnyType;
      case 1: return AnyShaped;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "inputs";
      case 1: return "outputs";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return AnyRankedTensor;
    }
    assert(false && "Invalid result index");
  }
};

class linalg_matvec : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 2; }
  unsigned getNumAttributes() const override { return 0; }
  unsigned getNumRegions() const override { return 1; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return AnyType;
      case 1: return AnyShaped;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "inputs";
      case 1: return "outputs";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return AnyRankedTensor;
    }
    assert(false && "Invalid result index");
  }
};

class linalg_mmt4d : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 2; }
  unsigned getNumAttributes() const override { return 0; }
  unsigned getNumRegions() const override { return 1; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return AnyType;
      case 1: return AnyShaped;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "inputs";
      case 1: return "outputs";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return AnyRankedTensor;
    }
    assert(false && "Invalid result index");
  }
};

class linalg_pooling_nchw_max : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 2; }
  unsigned getNumAttributes() const override { return 2; }
  unsigned getNumRegions() const override { return 1; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return AnyType;
      case 1: return AnyShaped;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "inputs";
      case 1: return "outputs";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return ::mlir::DenseIntElementsAttr();
      case 1: return ::mlir::DenseIntElementsAttr();
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
      case 0: return "strides";
      case 1: return "dilations";
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
      case 0: return false;
      case 1: return false;
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    // attrs.push_back(attrGen->genDenseIntElementsAttr("strides"));
    // attrs.push_back(attrGen->genDenseIntElementsAttr("dilations"));
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return AnyRankedTensor;
    }
    assert(false && "Invalid result index");
  }
};

class linalg_pooling_nchw_sum : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 2; }
  unsigned getNumAttributes() const override { return 2; }
  unsigned getNumRegions() const override { return 1; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return AnyType;
      case 1: return AnyShaped;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "inputs";
      case 1: return "outputs";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return ::mlir::DenseIntElementsAttr();
      case 1: return ::mlir::DenseIntElementsAttr();
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
      case 0: return "strides";
      case 1: return "dilations";
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
      case 0: return false;
      case 1: return false;
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    // attrs.push_back(attrGen->genDenseIntElementsAttr("strides"));
    // attrs.push_back(attrGen->genDenseIntElementsAttr("dilations"));
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return AnyRankedTensor;
    }
    assert(false && "Invalid result index");
  }
};

class linalg_pooling_ndhwc_max : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 2; }
  unsigned getNumAttributes() const override { return 2; }
  unsigned getNumRegions() const override { return 1; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return AnyType;
      case 1: return AnyShaped;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "inputs";
      case 1: return "outputs";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return ::mlir::DenseIntElementsAttr();
      case 1: return ::mlir::DenseIntElementsAttr();
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
      case 0: return "strides";
      case 1: return "dilations";
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
      case 0: return false;
      case 1: return false;
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    // attrs.push_back(attrGen->genDenseIntElementsAttr("strides"));
    // attrs.push_back(attrGen->genDenseIntElementsAttr("dilations"));
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return AnyRankedTensor;
    }
    assert(false && "Invalid result index");
  }
};

class linalg_pooling_ndhwc_min : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 2; }
  unsigned getNumAttributes() const override { return 2; }
  unsigned getNumRegions() const override { return 1; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return AnyType;
      case 1: return AnyShaped;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "inputs";
      case 1: return "outputs";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return ::mlir::DenseIntElementsAttr();
      case 1: return ::mlir::DenseIntElementsAttr();
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
      case 0: return "strides";
      case 1: return "dilations";
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
      case 0: return false;
      case 1: return false;
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    // attrs.push_back(attrGen->genDenseIntElementsAttr("strides"));
    // attrs.push_back(attrGen->genDenseIntElementsAttr("dilations"));
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return AnyRankedTensor;
    }
    assert(false && "Invalid result index");
  }
};

class linalg_pooling_ndhwc_sum : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 2; }
  unsigned getNumAttributes() const override { return 2; }
  unsigned getNumRegions() const override { return 1; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return AnyType;
      case 1: return AnyShaped;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "inputs";
      case 1: return "outputs";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return ::mlir::DenseIntElementsAttr();
      case 1: return ::mlir::DenseIntElementsAttr();
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
      case 0: return "strides";
      case 1: return "dilations";
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
      case 0: return false;
      case 1: return false;
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    // attrs.push_back(attrGen->genDenseIntElementsAttr("strides"));
    // attrs.push_back(attrGen->genDenseIntElementsAttr("dilations"));
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return AnyRankedTensor;
    }
    assert(false && "Invalid result index");
  }
};

class linalg_pooling_nhwc_max : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 2; }
  unsigned getNumAttributes() const override { return 2; }
  unsigned getNumRegions() const override { return 1; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return AnyType;
      case 1: return AnyShaped;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "inputs";
      case 1: return "outputs";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return ::mlir::DenseIntElementsAttr();
      case 1: return ::mlir::DenseIntElementsAttr();
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
      case 0: return "strides";
      case 1: return "dilations";
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
      case 0: return false;
      case 1: return false;
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    // attrs.push_back(attrGen->genDenseIntElementsAttr("strides"));
    // attrs.push_back(attrGen->genDenseIntElementsAttr("dilations"));
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return AnyRankedTensor;
    }
    assert(false && "Invalid result index");
  }
};

class linalg_pooling_nhwc_max_unsigned : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 2; }
  unsigned getNumAttributes() const override { return 2; }
  unsigned getNumRegions() const override { return 1; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return AnyType;
      case 1: return AnyShaped;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "inputs";
      case 1: return "outputs";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return ::mlir::DenseIntElementsAttr();
      case 1: return ::mlir::DenseIntElementsAttr();
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
      case 0: return "strides";
      case 1: return "dilations";
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
      case 0: return false;
      case 1: return false;
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    // attrs.push_back(attrGen->genDenseIntElementsAttr("strides"));
    // attrs.push_back(attrGen->genDenseIntElementsAttr("dilations"));
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return AnyRankedTensor;
    }
    assert(false && "Invalid result index");
  }
};

class linalg_pooling_nhwc_min : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 2; }
  unsigned getNumAttributes() const override { return 2; }
  unsigned getNumRegions() const override { return 1; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return AnyType;
      case 1: return AnyShaped;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "inputs";
      case 1: return "outputs";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return ::mlir::DenseIntElementsAttr();
      case 1: return ::mlir::DenseIntElementsAttr();
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
      case 0: return "strides";
      case 1: return "dilations";
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
      case 0: return false;
      case 1: return false;
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    // attrs.push_back(attrGen->genDenseIntElementsAttr("strides"));
    // attrs.push_back(attrGen->genDenseIntElementsAttr("dilations"));
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return AnyRankedTensor;
    }
    assert(false && "Invalid result index");
  }
};

class linalg_pooling_nhwc_min_unsigned : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 2; }
  unsigned getNumAttributes() const override { return 2; }
  unsigned getNumRegions() const override { return 1; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return AnyType;
      case 1: return AnyShaped;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "inputs";
      case 1: return "outputs";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return ::mlir::DenseIntElementsAttr();
      case 1: return ::mlir::DenseIntElementsAttr();
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
      case 0: return "strides";
      case 1: return "dilations";
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
      case 0: return false;
      case 1: return false;
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    // attrs.push_back(attrGen->genDenseIntElementsAttr("strides"));
    // attrs.push_back(attrGen->genDenseIntElementsAttr("dilations"));
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return AnyRankedTensor;
    }
    assert(false && "Invalid result index");
  }
};

class linalg_pooling_nhwc_sum : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 2; }
  unsigned getNumAttributes() const override { return 2; }
  unsigned getNumRegions() const override { return 1; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return AnyType;
      case 1: return AnyShaped;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "inputs";
      case 1: return "outputs";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return ::mlir::DenseIntElementsAttr();
      case 1: return ::mlir::DenseIntElementsAttr();
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
      case 0: return "strides";
      case 1: return "dilations";
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
      case 0: return false;
      case 1: return false;
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    // attrs.push_back(attrGen->genDenseIntElementsAttr("strides"));
    // attrs.push_back(attrGen->genDenseIntElementsAttr("dilations"));
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return AnyRankedTensor;
    }
    assert(false && "Invalid result index");
  }
};

class linalg_quantized_batch_matmul : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 2; }
  unsigned getNumAttributes() const override { return 0; }
  unsigned getNumRegions() const override { return 1; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return AnyType;
      case 1: return AnyShaped;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "inputs";
      case 1: return "outputs";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return AnyRankedTensor;
    }
    assert(false && "Invalid result index");
  }
};

class linalg_quantized_matmul : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 2; }
  unsigned getNumAttributes() const override { return 0; }
  unsigned getNumRegions() const override { return 1; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return AnyType;
      case 1: return AnyShaped;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "inputs";
      case 1: return "outputs";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return AnyRankedTensor;
    }
    assert(false && "Invalid result index");
  }
};

class linalg_reduce : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 2; }
  unsigned getNumAttributes() const override { return 1; }
  unsigned getNumRegions() const override { return 1; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return TensorOrMemref;
      case 1: return TensorOrMemref;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "inputs";
      case 1: return "inits";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return ::mlir::DenseI64ArrayAttr();
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
      case 0: return "dimensions";
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
      case 0: return true;
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    attrs.push_back(attrGen->genDenseI64ArrayAttr("dimensions"));
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return AnyTensor;
    }
    assert(false && "Invalid result index");
  }
};

class stablehlo_abs : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 1; }
  unsigned getNumAttributes() const override { return 0; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return anonymous_610;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "operand";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return anonymous_610;
    }
    assert(false && "Invalid result index");
  }
};

class stablehlo_add : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 2; }
  unsigned getNumAttributes() const override { return 0; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
      case 1: return HLO_Tensor;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "lhs";
      case 1: return "rhs";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class stablehlo_after_all : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 1; }
  unsigned getNumAttributes() const override { return 0; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Token;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "inputs";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Token;
    }
    assert(false && "Invalid result index");
  }
};

class stablehlo_all_gather : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 1; }
  unsigned getNumAttributes() const override { return 4; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "operand";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return ::mlir::IntegerAttr();
      case 1: return ::mlir::DenseIntElementsAttr();
      case 2: return ::mlir::stablehlo::ChannelHandleAttr();
      case 3: return ::mlir::UnitAttr();
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
      case 0: return "all_gather_dim";
      case 1: return "replica_groups";
      case 2: return "channel_handle";
      case 3: return "use_global_device_ids";
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
      case 0: return true;
      case 1: return true;
      case 2: return false;
      case 3: return false;
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    attrs.push_back(attrGen->genIntegerAttr("all_gather_dim"));
    attrs.push_back(attrGen->genDenseIntElementsAttr("replica_groups"));
    // attrs.push_back(attrGen->genChannelHandleAttr("channel_handle"));
    // attrs.push_back(attrGen->genUnitAttr("use_global_device_ids"));
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class stablehlo_all_reduce : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 1; }
  unsigned getNumAttributes() const override { return 3; }
  unsigned getNumRegions() const override { return 1; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "operand";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return ::mlir::DenseIntElementsAttr();
      case 1: return ::mlir::stablehlo::ChannelHandleAttr();
      case 2: return ::mlir::UnitAttr();
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
      case 0: return "replica_groups";
      case 1: return "channel_handle";
      case 2: return "use_global_device_ids";
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
      case 0: return true;
      case 1: return false;
      case 2: return false;
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    attrs.push_back(attrGen->genDenseIntElementsAttr("replica_groups"));
    // attrs.push_back(attrGen->genChannelHandleAttr("channel_handle"));
    // attrs.push_back(attrGen->genUnitAttr("use_global_device_ids"));
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class stablehlo_all_to_all : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 1; }
  unsigned getNumAttributes() const override { return 4; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "operand";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return ::mlir::IntegerAttr();
      case 1: return ::mlir::IntegerAttr();
      case 2: return ::mlir::IntegerAttr();
      case 3: return ::mlir::DenseIntElementsAttr();
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
      case 0: return "split_dimension";
      case 1: return "concat_dimension";
      case 2: return "split_count";
      case 3: return "replica_groups";
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
      case 0: return true;
      case 1: return true;
      case 2: return true;
      case 3: return true;
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    attrs.push_back(attrGen->genIntegerAttr("split_dimension"));
    attrs.push_back(attrGen->genIntegerAttr("concat_dimension"));
    attrs.push_back(attrGen->genIntegerAttr("split_count"));
    attrs.push_back(attrGen->genDenseIntElementsAttr("replica_groups"));
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class stablehlo_and : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 2; }
  unsigned getNumAttributes() const override { return 0; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_PredOrIntTensor;
      case 1: return HLO_PredOrIntTensor;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "lhs";
      case 1: return "rhs";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class stablehlo_atan2 : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 2; }
  unsigned getNumAttributes() const override { return 0; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
      case 1: return HLO_Tensor;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "lhs";
      case 1: return "rhs";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class stablehlo_batch_norm_grad : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 5; }
  unsigned getNumAttributes() const override { return 2; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 3; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return anonymous_653;
      case 1: return anonymous_655;
      case 2: return anonymous_655;
      case 3: return anonymous_655;
      case 4: return anonymous_653;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "operand";
      case 1: return "scale";
      case 2: return "mean";
      case 3: return "variance";
      case 4: return "grad_output";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return ::mlir::FloatAttr();
      case 1: return ::mlir::IntegerAttr();
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
      case 0: return "epsilon";
      case 1: return "feature_index";
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
      case 0: return true;
      case 1: return true;
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    attrs.push_back(attrGen->genFloatAttr("epsilon"));
    attrs.push_back(attrGen->genIntegerAttr("feature_index"));
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return anonymous_653;
      case 1: return anonymous_655;
      case 2: return anonymous_655;
    }
    assert(false && "Invalid result index");
  }
};

class stablehlo_batch_norm_inference : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 5; }
  unsigned getNumAttributes() const override { return 2; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return anonymous_653;
      case 1: return anonymous_655;
      case 2: return anonymous_655;
      case 3: return anonymous_655;
      case 4: return anonymous_655;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "operand";
      case 1: return "scale";
      case 2: return "offset";
      case 3: return "mean";
      case 4: return "variance";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return ::mlir::FloatAttr();
      case 1: return ::mlir::IntegerAttr();
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
      case 0: return "epsilon";
      case 1: return "feature_index";
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
      case 0: return true;
      case 1: return true;
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    attrs.push_back(attrGen->genFloatAttr("epsilon"));
    attrs.push_back(attrGen->genIntegerAttr("feature_index"));
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return anonymous_653;
    }
    assert(false && "Invalid result index");
  }
};

class stablehlo_batch_norm_training : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 3; }
  unsigned getNumAttributes() const override { return 2; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 3; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return anonymous_653;
      case 1: return anonymous_655;
      case 2: return anonymous_655;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "operand";
      case 1: return "scale";
      case 2: return "offset";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return ::mlir::FloatAttr();
      case 1: return ::mlir::IntegerAttr();
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
      case 0: return "epsilon";
      case 1: return "feature_index";
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
      case 0: return true;
      case 1: return true;
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    attrs.push_back(attrGen->genFloatAttr("epsilon"));
    attrs.push_back(attrGen->genIntegerAttr("feature_index"));
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return anonymous_653;
      case 1: return anonymous_655;
      case 2: return anonymous_655;
    }
    assert(false && "Invalid result index");
  }
};

class stablehlo_bitcast_convert : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 1; }
  unsigned getNumAttributes() const override { return 0; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "operand";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class stablehlo_broadcast_in_dim : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 1; }
  unsigned getNumAttributes() const override { return 1; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "operand";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return ::mlir::DenseIntElementsAttr();
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
      case 0: return "broadcast_dimensions";
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
      case 0: return true;
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    attrs.push_back(attrGen->genDenseIntElementsAttr("broadcast_dimensions"));
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_StaticShapeTensor;
    }
    assert(false && "Invalid result index");
  }
};

class stablehlo_broadcast : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 1; }
  unsigned getNumAttributes() const override { return 1; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "operand";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return ::mlir::DenseIntElementsAttr();
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
      case 0: return "broadcast_sizes";
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
      case 0: return true;
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    attrs.push_back(attrGen->genDenseIntElementsAttr("broadcast_sizes"));
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class stablehlo_case : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 1; }
  unsigned getNumAttributes() const override { return 0; }
  unsigned getNumRegions() const override { return 1; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return I32Tensor;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "index";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_TensorOrToken;
    }
    assert(false && "Invalid result index");
  }
};

class stablehlo_cbrt : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 1; }
  unsigned getNumAttributes() const override { return 0; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpTensor;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "operand";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpTensor;
    }
    assert(false && "Invalid result index");
  }
};

class stablehlo_ceil : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 1; }
  unsigned getNumAttributes() const override { return 0; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpTensor;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "operand";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpTensor;
    }
    assert(false && "Invalid result index");
  }
};

class stablehlo_cholesky : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 1; }
  unsigned getNumAttributes() const override { return 1; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpOrComplexTensor;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "a";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return ::mlir::BoolAttr();
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
      case 0: return "lower";
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
      case 0: return false;
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    // attrs.push_back(attrGen->genBoolAttr("lower"));
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpOrComplexTensor;
    }
    assert(false && "Invalid result index");
  }
};

class stablehlo_clamp : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 3; }
  unsigned getNumAttributes() const override { return 0; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
      case 1: return HLO_Tensor;
      case 2: return HLO_Tensor;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "min";
      case 1: return "operand";
      case 2: return "max";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class stablehlo_count_leading_zeros : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 1; }
  unsigned getNumAttributes() const override { return 0; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_IntTensor;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "operand";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_IntTensor;
    }
    assert(false && "Invalid result index");
  }
};

class stablehlo_collective_permute : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 1; }
  unsigned getNumAttributes() const override { return 2; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "operand";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return ::mlir::DenseIntElementsAttr();
      case 1: return ::mlir::stablehlo::ChannelHandleAttr();
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
      case 0: return "source_target_pairs";
      case 1: return "channel_handle";
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
      case 0: return true;
      case 1: return false;
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    attrs.push_back(attrGen->genDenseIntElementsAttr("source_target_pairs"));
    // attrs.push_back(attrGen->genChannelHandleAttr("channel_handle"));
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class stablehlo_compare : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 2; }
  unsigned getNumAttributes() const override { return 2; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
      case 1: return HLO_Tensor;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "lhs";
      case 1: return "rhs";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return ::mlir::stablehlo::ComparisonDirectionAttr();
      case 1: return ::mlir::stablehlo::ComparisonTypeAttr();
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
      case 0: return "comparison_direction";
      case 1: return "compare_type";
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
      case 0: return true;
      case 1: return false;
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    attrs.push_back(attrGen->genStablehloComparisonDirectionAttr("comparison_direction"));
    // attrs.push_back(attrGen->genComparisonTypeAttr("compare_type"));
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_PredTensor;
    }
    assert(false && "Invalid result index");
  }
};

class stablehlo_complex : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 2; }
  unsigned getNumAttributes() const override { return 0; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Fp32Or64Tensor;
      case 1: return HLO_Fp32Or64Tensor;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "lhs";
      case 1: return "rhs";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_ComplexTensor;
    }
    assert(false && "Invalid result index");
  }
};

class stablehlo_compute_reshape_shape : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 2; }
  unsigned getNumAttributes() const override { return 0; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return Index;
      case 1: return anonymous_704;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "num_elements";
      case 1: return "dynamic_shape";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return anonymous_704;
    }
    assert(false && "Invalid result index");
  }
};

class stablehlo_concatenate : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 1; }
  unsigned getNumAttributes() const override { return 1; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "inputs";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return ::mlir::IntegerAttr();
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
      case 0: return "dimension";
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
      case 0: return true;
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    attrs.push_back(attrGen->genIntegerAttr("dimension"));
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class stablehlo_constant : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 0; }
  unsigned getNumAttributes() const override { return 1; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return ::mlir::ElementsAttr();
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
      case 0: return "value";
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
      case 0: return true;
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    attrs.push_back(attrGen->genElementsAttr("value"));
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_StaticShapeTensor;
    }
    assert(false && "Invalid result index");
  }
};

class stablehlo_convert : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 1; }
  unsigned getNumAttributes() const override { return 0; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "operand";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class stablehlo_convolution : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 2; }
  unsigned getNumAttributes() const override { return 9; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
      case 1: return HLO_Tensor;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "lhs";
      case 1: return "rhs";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return ::mlir::DenseIntElementsAttr();
      case 1: return ::mlir::DenseIntElementsAttr();
      case 2: return ::mlir::DenseIntElementsAttr();
      case 3: return ::mlir::DenseIntElementsAttr();
      case 4: return ::mlir::DenseElementsAttr();
      case 5: return ::mlir::stablehlo::ConvDimensionNumbersAttr();
      case 6: return ::mlir::IntegerAttr();
      case 7: return ::mlir::IntegerAttr();
      case 8: return ::mlir::ArrayAttr();
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
      case 0: return "window_strides";
      case 1: return "padding";
      case 2: return "lhs_dilation";
      case 3: return "rhs_dilation";
      case 4: return "window_reversal";
      case 5: return "dimension_numbers";
      case 6: return "feature_group_count";
      case 7: return "batch_group_count";
      case 8: return "precision_config";
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
      case 0: return false;
      case 1: return false;
      case 2: return false;
      case 3: return false;
      case 4: return false;
      case 5: return true;
      case 6: return true;
      case 7: return true;
      case 8: return false;
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    // attrs.push_back(attrGen->genDenseIntElementsAttr("window_strides"));
    // attrs.push_back(attrGen->genDenseIntElementsAttr("padding"));
    // attrs.push_back(attrGen->genDenseIntElementsAttr("lhs_dilation"));
    // attrs.push_back(attrGen->genDenseIntElementsAttr("rhs_dilation"));
    // attrs.push_back(attrGen->genDenseElementsAttr("window_reversal"));
    attrs.push_back(attrGen->genStablehloConvDimensionNumbersAttr("dimension_numbers"));
    attrs.push_back(attrGen->genIntegerAttr("feature_group_count"));
    attrs.push_back(attrGen->genIntegerAttr("batch_group_count"));
    // attrs.push_back(attrGen->genArrayAttr("precision_config"));
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class stablehlo_cosine : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 1; }
  unsigned getNumAttributes() const override { return 0; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpOrComplexTensor;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "operand";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpOrComplexTensor;
    }
    assert(false && "Invalid result index");
  }
};

class stablehlo_create_token : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 0; }
  unsigned getNumAttributes() const override { return 0; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Token;
    }
    assert(false && "Invalid result index");
  }
};

class stablehlo_cross_replica_sum : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 1; }
  unsigned getNumAttributes() const override { return 1; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "operand";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return ::mlir::DenseIntElementsAttr();
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
      case 0: return "replica_groups";
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
      case 0: return true;
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    attrs.push_back(attrGen->genDenseIntElementsAttr("replica_groups"));
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class stablehlo_cstr_reshapable : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 2; }
  unsigned getNumAttributes() const override { return 0; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return Index;
      case 1: return anonymous_704;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "num_elements";
      case 1: return "dynamic_shape";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return Shape_WitnessType;
    }
    assert(false && "Invalid result index");
  }
};

class stablehlo_custom_call : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 1; }
  unsigned getNumAttributes() const override { return 7; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_TensorOrTokenOrTuple;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "inputs";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return ::mlir::StringAttr();
      case 1: return ::mlir::BoolAttr();
      case 2: return ::mlir::StringAttr();
      case 3: return ::mlir::stablehlo::CustomCallApiVersionAttr();
      case 4: return ::mlir::ArrayAttr();
      case 5: return ::mlir::ArrayAttr();
      case 6: return ::mlir::ArrayAttr();
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
      case 0: return "call_target_name";
      case 1: return "has_side_effect";
      case 2: return "backend_config";
      case 3: return "api_version";
      case 4: return "called_computations";
      case 5: return "operand_layouts";
      case 6: return "result_layouts";
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
      case 0: return true;
      case 1: return false;
      case 2: return true;
      case 3: return false;
      case 4: return false;
      case 5: return false;
      case 6: return false;
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    attrs.push_back(attrGen->genStringAttr("call_target_name"));
    // attrs.push_back(attrGen->genBoolAttr("has_side_effect"));
    attrs.push_back(attrGen->genStringAttr("backend_config"));
    // attrs.push_back(attrGen->genCustomCallApiVersionAttr("api_version"));
    // attrs.push_back(attrGen->genArrayAttr("called_computations"));
    // attrs.push_back(attrGen->genArrayAttr("operand_layouts"));
    // attrs.push_back(attrGen->genArrayAttr("result_layouts"));
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_TensorOrTokenOrTuple;
    }
    assert(false && "Invalid result index");
  }
};

class stablehlo_divide : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 2; }
  unsigned getNumAttributes() const override { return 0; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
      case 1: return HLO_Tensor;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "lhs";
      case 1: return "rhs";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class stablehlo_dot_general : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 2; }
  unsigned getNumAttributes() const override { return 2; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
      case 1: return HLO_Tensor;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "lhs";
      case 1: return "rhs";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return ::mlir::stablehlo::DotDimensionNumbersAttr();
      case 1: return ::mlir::ArrayAttr();
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
      case 0: return "dot_dimension_numbers";
      case 1: return "precision_config";
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
      case 0: return true;
      case 1: return false;
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    attrs.push_back(attrGen->genStablehloDotDimensionNumbersAttr("dot_dimension_numbers"));
    // attrs.push_back(attrGen->genArrayAttr("precision_config"));
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class stablehlo_dot : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 2; }
  unsigned getNumAttributes() const override { return 1; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
      case 1: return HLO_Tensor;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "lhs";
      case 1: return "rhs";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return ::mlir::ArrayAttr();
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
      case 0: return "precision_config";
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
      case 0: return false;
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    // attrs.push_back(attrGen->genArrayAttr("precision_config"));
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class stablehlo_dynamic_broadcast_in_dim : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 2; }
  unsigned getNumAttributes() const override { return 3; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
      case 1: return HLO_DimensionTensor;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "operand";
      case 1: return "output_dimensions";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return ::mlir::DenseIntElementsAttr();
      case 1: return ::mlir::DenseIntElementsAttr();
      case 2: return ::mlir::DenseIntElementsAttr();
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
      case 0: return "broadcast_dimensions";
      case 1: return "known_expanding_dimensions";
      case 2: return "known_nonexpanding_dimensions";
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
      case 0: return true;
      case 1: return false;
      case 2: return false;
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    attrs.push_back(attrGen->genDenseIntElementsAttr("broadcast_dimensions"));
    // attrs.push_back(attrGen->genDenseIntElementsAttr("known_expanding_dimensions"));
    // attrs.push_back(attrGen->genDenseIntElementsAttr("known_nonexpanding_dimensions"));
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class stablehlo_dynamic_conv : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 3; }
  unsigned getNumAttributes() const override { return 9; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
      case 1: return HLO_Tensor;
      case 2: return HLO_Tensor;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "lhs";
      case 1: return "rhs";
      case 2: return "d_padding";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return ::mlir::DenseIntElementsAttr();
      case 1: return ::mlir::DenseIntElementsAttr();
      case 2: return ::mlir::DenseIntElementsAttr();
      case 3: return ::mlir::DenseIntElementsAttr();
      case 4: return ::mlir::DenseElementsAttr();
      case 5: return ::mlir::stablehlo::ConvDimensionNumbersAttr();
      case 6: return ::mlir::IntegerAttr();
      case 7: return ::mlir::IntegerAttr();
      case 8: return ::mlir::ArrayAttr();
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
      case 0: return "window_strides";
      case 1: return "padding";
      case 2: return "lhs_dilation";
      case 3: return "rhs_dilation";
      case 4: return "window_reversal";
      case 5: return "dimension_numbers";
      case 6: return "feature_group_count";
      case 7: return "batch_group_count";
      case 8: return "precision_config";
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
      case 0: return false;
      case 1: return false;
      case 2: return false;
      case 3: return false;
      case 4: return false;
      case 5: return true;
      case 6: return true;
      case 7: return true;
      case 8: return false;
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    // attrs.push_back(attrGen->genDenseIntElementsAttr("window_strides"));
    // attrs.push_back(attrGen->genDenseIntElementsAttr("padding"));
    // attrs.push_back(attrGen->genDenseIntElementsAttr("lhs_dilation"));
    // attrs.push_back(attrGen->genDenseIntElementsAttr("rhs_dilation"));
    // attrs.push_back(attrGen->genDenseElementsAttr("window_reversal"));
    attrs.push_back(attrGen->genStablehloConvDimensionNumbersAttr("dimension_numbers"));
    attrs.push_back(attrGen->genIntegerAttr("feature_group_count"));
    attrs.push_back(attrGen->genIntegerAttr("batch_group_count"));
    // attrs.push_back(attrGen->genArrayAttr("precision_config"));
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class stablehlo_dynamic_gather : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 3; }
  unsigned getNumAttributes() const override { return 2; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
      case 1: return HLO_IntTensor;
      case 2: return HLO_IntTensor;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "operand";
      case 1: return "start_indices";
      case 2: return "slice_sizes";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return ::mlir::stablehlo::GatherDimensionNumbersAttr();
      case 1: return ::mlir::BoolAttr();
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
      case 0: return "dimension_numbers";
      case 1: return "indices_are_sorted";
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
      case 0: return true;
      case 1: return false;
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    attrs.push_back(attrGen->genStablehloGatherDimensionNumbersAttr("dimension_numbers"));
    // attrs.push_back(attrGen->genBoolAttr("indices_are_sorted"));
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class stablehlo_dynamic_iota : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 1; }
  unsigned getNumAttributes() const override { return 1; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_DimensionTensor;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "output_shape";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return ::mlir::IntegerAttr();
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
      case 0: return "iota_dimension";
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
      case 0: return true;
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    attrs.push_back(attrGen->genIntegerAttr("iota_dimension"));
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class stablehlo_dynamic_pad : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 5; }
  unsigned getNumAttributes() const override { return 0; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
      case 1: return HLO_Tensor;
      case 2: return HLO_DimensionTensor;
      case 3: return HLO_DimensionTensor;
      case 4: return HLO_DimensionTensor;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "operand";
      case 1: return "padding_value";
      case 2: return "edge_padding_low";
      case 3: return "edge_padding_high";
      case 4: return "interior_padding";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class stablehlo_dynamic_reshape : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 2; }
  unsigned getNumAttributes() const override { return 0; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
      case 1: return HLO_DimensionTensor;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "operand";
      case 1: return "output_shape";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class stablehlo_dynamic_slice : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 2; }
  unsigned getNumAttributes() const override { return 1; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
      case 1: return HLO_ScalarIntTensor;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "operand";
      case 1: return "start_indices";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return ::mlir::DenseIntElementsAttr();
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
      case 0: return "slice_sizes";
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
      case 0: return true;
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    attrs.push_back(attrGen->genDenseIntElementsAttr("slice_sizes"));
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class stablehlo_dynamic_update_slice : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 3; }
  unsigned getNumAttributes() const override { return 0; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
      case 1: return HLO_Tensor;
      case 2: return HLO_ScalarIntTensor;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "operand";
      case 1: return "update";
      case 2: return "start_indices";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class stablehlo_einsum : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 2; }
  unsigned getNumAttributes() const override { return 1; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
      case 1: return HLO_Tensor;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "lhs";
      case 1: return "rhs";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return ::mlir::StringAttr();
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
      case 0: return "einsum_config";
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
      case 0: return true;
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    attrs.push_back(attrGen->genStringAttr("einsum_config"));
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class stablehlo_exponential : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 1; }
  unsigned getNumAttributes() const override { return 0; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpOrComplexTensor;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "operand";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpOrComplexTensor;
    }
    assert(false && "Invalid result index");
  }
};

class stablehlo_exponential_minus_one : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 1; }
  unsigned getNumAttributes() const override { return 0; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpOrComplexTensor;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "operand";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpOrComplexTensor;
    }
    assert(false && "Invalid result index");
  }
};

class stablehlo_fft : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 1; }
  unsigned getNumAttributes() const override { return 2; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "operand";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return ::mlir::stablehlo::FftTypeAttr();
      case 1: return ::mlir::DenseIntElementsAttr();
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
      case 0: return "fft_type";
      case 1: return "fft_length";
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
      case 0: return true;
      case 1: return true;
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    attrs.push_back(attrGen->genStablehloFftTypeAttr("fft_type"));
    attrs.push_back(attrGen->genDenseIntElementsAttr("fft_length"));
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class stablehlo_floor : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 1; }
  unsigned getNumAttributes() const override { return 0; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpTensor;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "operand";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpTensor;
    }
    assert(false && "Invalid result index");
  }
};

class stablehlo_gather : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 2; }
  unsigned getNumAttributes() const override { return 3; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
      case 1: return HLO_IntTensor;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "operand";
      case 1: return "start_indices";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return ::mlir::stablehlo::GatherDimensionNumbersAttr();
      case 1: return ::mlir::DenseIntElementsAttr();
      case 2: return ::mlir::BoolAttr();
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
      case 0: return "dimension_numbers";
      case 1: return "slice_sizes";
      case 2: return "indices_are_sorted";
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
      case 0: return true;
      case 1: return true;
      case 2: return false;
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    attrs.push_back(attrGen->genStablehloGatherDimensionNumbersAttr("dimension_numbers"));
    attrs.push_back(attrGen->genDenseIntElementsAttr("slice_sizes"));
    // attrs.push_back(attrGen->genBoolAttr("indices_are_sorted"));
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class stablehlo_get_dimension_size : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 1; }
  unsigned getNumAttributes() const override { return 1; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "operand";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return ::mlir::IntegerAttr();
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
      case 0: return "dimension";
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
      case 0: return true;
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    attrs.push_back(attrGen->genIntegerAttr("dimension"));
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return I32Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class stablehlo_get_tuple_element : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 1; }
  unsigned getNumAttributes() const override { return 1; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tuple;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "operand";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return ::mlir::IntegerAttr();
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
      case 0: return "index";
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
      case 0: return true;
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    attrs.push_back(attrGen->genIntegerAttr("index"));
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_TensorOrTokenOrTuple;
    }
    assert(false && "Invalid result index");
  }
};

class stablehlo_if : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 1; }
  unsigned getNumAttributes() const override { return 0; }
  unsigned getNumRegions() const override { return 2; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_PredTensor;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "pred";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_TensorOrToken;
    }
    assert(false && "Invalid result index");
  }
};

class stablehlo_imag : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 1; }
  unsigned getNumAttributes() const override { return 0; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpOrComplexTensor;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "operand";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpTensor;
    }
    assert(false && "Invalid result index");
  }
};

class stablehlo_infeed : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 1; }
  unsigned getNumAttributes() const override { return 2; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Token;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "token";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return ::mlir::StringAttr();
      case 1: return ::mlir::ArrayAttr();
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
      case 0: return "infeed_config";
      case 1: return "layout";
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
      case 0: return true;
      case 1: return false;
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    attrs.push_back(attrGen->genStringAttr("infeed_config"));
    // attrs.push_back(attrGen->genArrayAttr("layout"));
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_TensorOrToken;
    }
    assert(false && "Invalid result index");
  }
};

class stablehlo_iota : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 0; }
  unsigned getNumAttributes() const override { return 1; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return ::mlir::IntegerAttr();
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
      case 0: return "iota_dimension";
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
      case 0: return true;
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    attrs.push_back(attrGen->genIntegerAttr("iota_dimension"));
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_IntFpOrComplexTensor;
    }
    assert(false && "Invalid result index");
  }
};

class stablehlo_is_finite : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 1; }
  unsigned getNumAttributes() const override { return 0; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpTensor;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "x";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_PredTensor;
    }
    assert(false && "Invalid result index");
  }
};

class stablehlo_log_plus_one : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 1; }
  unsigned getNumAttributes() const override { return 0; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpOrComplexTensor;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "operand";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpOrComplexTensor;
    }
    assert(false && "Invalid result index");
  }
};

class stablehlo_log : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 1; }
  unsigned getNumAttributes() const override { return 0; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpOrComplexTensor;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "operand";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpOrComplexTensor;
    }
    assert(false && "Invalid result index");
  }
};

class stablehlo_logistic : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 1; }
  unsigned getNumAttributes() const override { return 0; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpOrComplexTensor;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "operand";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpOrComplexTensor;
    }
    assert(false && "Invalid result index");
  }
};

class stablehlo_map : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 1; }
  unsigned getNumAttributes() const override { return 1; }
  unsigned getNumRegions() const override { return 1; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "inputs";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return ::mlir::DenseIntElementsAttr();
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
      case 0: return "dimensions";
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
      case 0: return true;
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    attrs.push_back(attrGen->genDenseIntElementsAttr("dimensions"));
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class stablehlo_maximum : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 2; }
  unsigned getNumAttributes() const override { return 0; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
      case 1: return HLO_Tensor;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "lhs";
      case 1: return "rhs";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class stablehlo_minimum : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 2; }
  unsigned getNumAttributes() const override { return 0; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
      case 1: return HLO_Tensor;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "lhs";
      case 1: return "rhs";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class stablehlo_multiply : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 2; }
  unsigned getNumAttributes() const override { return 0; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
      case 1: return HLO_Tensor;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "lhs";
      case 1: return "rhs";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class stablehlo_negate : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 1; }
  unsigned getNumAttributes() const override { return 0; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_IntFpOrComplexTensor;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "operand";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_IntFpOrComplexTensor;
    }
    assert(false && "Invalid result index");
  }
};

class stablehlo_not : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 1; }
  unsigned getNumAttributes() const override { return 0; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_PredOrIntTensor;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "operand";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_PredOrIntTensor;
    }
    assert(false && "Invalid result index");
  }
};

class stablehlo_optimization_barrier : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 1; }
  unsigned getNumAttributes() const override { return 0; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_TensorOrToken;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "operand";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_TensorOrToken;
    }
    assert(false && "Invalid result index");
  }
};

class stablehlo_or : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 2; }
  unsigned getNumAttributes() const override { return 0; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_PredOrIntTensor;
      case 1: return HLO_PredOrIntTensor;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "lhs";
      case 1: return "rhs";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class stablehlo_outfeed : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 2; }
  unsigned getNumAttributes() const override { return 1; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
      case 1: return HLO_Token;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "inputs";
      case 1: return "token";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return ::mlir::StringAttr();
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
      case 0: return "outfeed_config";
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
      case 0: return true;
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    attrs.push_back(attrGen->genStringAttr("outfeed_config"));
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Token;
    }
    assert(false && "Invalid result index");
  }
};

class stablehlo_pad : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 2; }
  unsigned getNumAttributes() const override { return 3; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
      case 1: return HLO_Tensor;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "operand";
      case 1: return "padding_value";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return ::mlir::DenseIntElementsAttr();
      case 1: return ::mlir::DenseIntElementsAttr();
      case 2: return ::mlir::DenseIntElementsAttr();
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
      case 0: return "edge_padding_low";
      case 1: return "edge_padding_high";
      case 2: return "interior_padding";
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
      case 0: return true;
      case 1: return true;
      case 2: return true;
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    attrs.push_back(attrGen->genDenseIntElementsAttr("edge_padding_low"));
    attrs.push_back(attrGen->genDenseIntElementsAttr("edge_padding_high"));
    attrs.push_back(attrGen->genDenseIntElementsAttr("interior_padding"));
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class stablehlo_popcnt : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 1; }
  unsigned getNumAttributes() const override { return 0; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_IntTensor;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "operand";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_IntTensor;
    }
    assert(false && "Invalid result index");
  }
};

class stablehlo_power : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 2; }
  unsigned getNumAttributes() const override { return 0; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
      case 1: return HLO_Tensor;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "lhs";
      case 1: return "rhs";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class stablehlo_real_dynamic_slice : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 4; }
  unsigned getNumAttributes() const override { return 0; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
      case 1: return HLO_DimensionTensor;
      case 2: return HLO_DimensionTensor;
      case 3: return HLO_DimensionTensor;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "operand";
      case 1: return "start_indices";
      case 2: return "limit_indices";
      case 3: return "strides";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class stablehlo_real : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 1; }
  unsigned getNumAttributes() const override { return 0; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpOrComplexTensor;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "operand";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpTensor;
    }
    assert(false && "Invalid result index");
  }
};

class stablehlo_recv : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 1; }
  unsigned getNumAttributes() const override { return 2; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Token;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "token";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return ::mlir::stablehlo::ChannelHandleAttr();
      case 1: return ::mlir::BoolAttr();
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
      case 0: return "channel_handle";
      case 1: return "is_host_transfer";
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
      case 0: return true;
      case 1: return false;
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    attrs.push_back(attrGen->genStablehloChannelHandleAttr("channel_handle"));
    // attrs.push_back(attrGen->genBoolAttr("is_host_transfer"));
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_TensorOrToken;
    }
    assert(false && "Invalid result index");
  }
};

class stablehlo_reduce : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 2; }
  unsigned getNumAttributes() const override { return 1; }
  unsigned getNumRegions() const override { return 1; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
      case 1: return HLO_Tensor;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "inputs";
      case 1: return "init_values";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return ::mlir::DenseIntElementsAttr();
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
      case 0: return "dimensions";
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
      case 0: return true;
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    attrs.push_back(attrGen->genDenseIntElementsAttr("dimensions"));
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class stablehlo_reduce_precision : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 1; }
  unsigned getNumAttributes() const override { return 2; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpTensor;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "operand";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return ::mlir::IntegerAttr();
      case 1: return ::mlir::IntegerAttr();
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
      case 0: return "exponent_bits";
      case 1: return "mantissa_bits";
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
      case 0: return true;
      case 1: return true;
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    attrs.push_back(attrGen->genIntegerAttr("exponent_bits"));
    attrs.push_back(attrGen->genIntegerAttr("mantissa_bits"));
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpTensor;
    }
    assert(false && "Invalid result index");
  }
};

class stablehlo_reduce_scatter : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 1; }
  unsigned getNumAttributes() const override { return 4; }
  unsigned getNumRegions() const override { return 1; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "operand";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return ::mlir::IntegerAttr();
      case 1: return ::mlir::DenseIntElementsAttr();
      case 2: return ::mlir::stablehlo::ChannelHandleAttr();
      case 3: return ::mlir::UnitAttr();
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
      case 0: return "scatter_dimension";
      case 1: return "replica_groups";
      case 2: return "channel_handle";
      case 3: return "use_global_device_ids";
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
      case 0: return true;
      case 1: return true;
      case 2: return false;
      case 3: return false;
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    attrs.push_back(attrGen->genIntegerAttr("scatter_dimension"));
    attrs.push_back(attrGen->genDenseIntElementsAttr("replica_groups"));
    // attrs.push_back(attrGen->genChannelHandleAttr("channel_handle"));
    // attrs.push_back(attrGen->genUnitAttr("use_global_device_ids"));
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class stablehlo_reduce_window : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 2; }
  unsigned getNumAttributes() const override { return 5; }
  unsigned getNumRegions() const override { return 1; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
      case 1: return HLO_Tensor;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "inputs";
      case 1: return "init_values";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return ::mlir::DenseIntElementsAttr();
      case 1: return ::mlir::DenseIntElementsAttr();
      case 2: return ::mlir::DenseIntElementsAttr();
      case 3: return ::mlir::DenseIntElementsAttr();
      case 4: return ::mlir::DenseIntElementsAttr();
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
      case 0: return "window_dimensions";
      case 1: return "window_strides";
      case 2: return "base_dilations";
      case 3: return "window_dilations";
      case 4: return "padding";
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
      case 0: return true;
      case 1: return false;
      case 2: return false;
      case 3: return false;
      case 4: return false;
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    attrs.push_back(attrGen->genDenseIntElementsAttr("window_dimensions"));
    // attrs.push_back(attrGen->genDenseIntElementsAttr("window_strides"));
    // attrs.push_back(attrGen->genDenseIntElementsAttr("base_dilations"));
    // attrs.push_back(attrGen->genDenseIntElementsAttr("window_dilations"));
    // attrs.push_back(attrGen->genDenseIntElementsAttr("padding"));
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class stablehlo_remainder : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 2; }
  unsigned getNumAttributes() const override { return 0; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
      case 1: return HLO_Tensor;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "lhs";
      case 1: return "rhs";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class stablehlo_replica_id : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 0; }
  unsigned getNumAttributes() const override { return 0; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return anonymous_621;
    }
    assert(false && "Invalid result index");
  }
};

class stablehlo_reshape : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 1; }
  unsigned getNumAttributes() const override { return 0; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "operand";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_StaticShapeTensor;
    }
    assert(false && "Invalid result index");
  }
};

class stablehlo_return : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 1; }
  unsigned getNumAttributes() const override { return 0; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 0; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_TensorOrTokenOrTuple;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "results";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid result index");
  }
};

class stablehlo_reverse : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 1; }
  unsigned getNumAttributes() const override { return 1; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "operand";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return ::mlir::DenseIntElementsAttr();
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
      case 0: return "dimensions";
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
      case 0: return true;
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    attrs.push_back(attrGen->genDenseIntElementsAttr("dimensions"));
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class stablehlo_rng_bit_generator : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 1; }
  unsigned getNumAttributes() const override { return 1; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 2; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_IntOrFpTensor;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "initial_state";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return ::mlir::stablehlo::RngAlgorithmAttr();
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
      case 0: return "rng_algorithm";
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
      case 0: return true;
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    attrs.push_back(attrGen->genStablehloRngAlgorithmAttr("rng_algorithm"));
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_IntOrFpTensor;
      case 1: return HLO_IntOrFpTensor;
    }
    assert(false && "Invalid result index");
  }
};

class stablehlo_rng : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 3; }
  unsigned getNumAttributes() const override { return 1; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return anonymous_686;
      case 1: return anonymous_686;
      case 2: return HLO_DimensionTensor;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "a";
      case 1: return "b";
      case 2: return "shape";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return ::mlir::stablehlo::RngDistributionAttr();
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
      case 0: return "rng_distribution";
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
      case 0: return true;
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    attrs.push_back(attrGen->genStablehloRngDistributionAttr("rng_distribution"));
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_PredIntOrFpTensor;
    }
    assert(false && "Invalid result index");
  }
};

class stablehlo_round_nearest_even : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 1; }
  unsigned getNumAttributes() const override { return 0; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpTensor;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "operand";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpTensor;
    }
    assert(false && "Invalid result index");
  }
};

class stablehlo_round_nearest_afz : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 1; }
  unsigned getNumAttributes() const override { return 0; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpTensor;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "operand";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpTensor;
    }
    assert(false && "Invalid result index");
  }
};

class stablehlo_rsqrt : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 1; }
  unsigned getNumAttributes() const override { return 0; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpOrComplexTensor;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "operand";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpOrComplexTensor;
    }
    assert(false && "Invalid result index");
  }
};

class stablehlo_scatter : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 3; }
  unsigned getNumAttributes() const override { return 3; }
  unsigned getNumRegions() const override { return 1; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
      case 1: return anonymous_673;
      case 2: return HLO_Tensor;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "inputs";
      case 1: return "scatter_indices";
      case 2: return "updates";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return ::mlir::stablehlo::ScatterDimensionNumbersAttr();
      case 1: return ::mlir::BoolAttr();
      case 2: return ::mlir::BoolAttr();
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
      case 0: return "scatter_dimension_numbers";
      case 1: return "indices_are_sorted";
      case 2: return "unique_indices";
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
      case 0: return true;
      case 1: return false;
      case 2: return false;
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    attrs.push_back(attrGen->genStablehloScatterDimensionNumbersAttr("scatter_dimension_numbers"));
    // attrs.push_back(attrGen->genBoolAttr("indices_are_sorted"));
    // attrs.push_back(attrGen->genBoolAttr("unique_indices"));
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class stablehlo_select_and_scatter : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 3; }
  unsigned getNumAttributes() const override { return 3; }
  unsigned getNumRegions() const override { return 2; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
      case 1: return HLO_Tensor;
      case 2: return HLO_Tensor;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "operand";
      case 1: return "source";
      case 2: return "init_value";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return ::mlir::DenseIntElementsAttr();
      case 1: return ::mlir::DenseIntElementsAttr();
      case 2: return ::mlir::DenseIntElementsAttr();
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
      case 0: return "window_dimensions";
      case 1: return "window_strides";
      case 2: return "padding";
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
      case 0: return false;
      case 1: return false;
      case 2: return false;
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    // attrs.push_back(attrGen->genDenseIntElementsAttr("window_dimensions"));
    // attrs.push_back(attrGen->genDenseIntElementsAttr("window_strides"));
    // attrs.push_back(attrGen->genDenseIntElementsAttr("padding"));
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class stablehlo_select : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 3; }
  unsigned getNumAttributes() const override { return 0; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_PredTensor;
      case 1: return HLO_Tensor;
      case 2: return HLO_Tensor;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "pred";
      case 1: return "on_true";
      case 2: return "on_false";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class stablehlo_send : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 2; }
  unsigned getNumAttributes() const override { return 2; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
      case 1: return HLO_Token;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "inputs";
      case 1: return "token";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return ::mlir::stablehlo::ChannelHandleAttr();
      case 1: return ::mlir::BoolAttr();
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
      case 0: return "channel_handle";
      case 1: return "is_host_transfer";
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
      case 0: return true;
      case 1: return false;
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    attrs.push_back(attrGen->genStablehloChannelHandleAttr("channel_handle"));
    // attrs.push_back(attrGen->genBoolAttr("is_host_transfer"));
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Token;
    }
    assert(false && "Invalid result index");
  }
};

class stablehlo_set_dimension_size : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 2; }
  unsigned getNumAttributes() const override { return 1; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
      case 1: return I32Tensor;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "operand";
      case 1: return "size";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return ::mlir::IntegerAttr();
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
      case 0: return "dimension";
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
      case 0: return true;
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    attrs.push_back(attrGen->genIntegerAttr("dimension"));
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class stablehlo_shift_left : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 2; }
  unsigned getNumAttributes() const override { return 0; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
      case 1: return HLO_Tensor;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "lhs";
      case 1: return "rhs";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class stablehlo_shift_right_arithmetic : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 2; }
  unsigned getNumAttributes() const override { return 0; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
      case 1: return HLO_Tensor;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "lhs";
      case 1: return "rhs";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class stablehlo_shift_right_logical : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 2; }
  unsigned getNumAttributes() const override { return 0; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
      case 1: return HLO_Tensor;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "lhs";
      case 1: return "rhs";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class stablehlo_sign : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 1; }
  unsigned getNumAttributes() const override { return 0; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return anonymous_610;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "operand";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return anonymous_610;
    }
    assert(false && "Invalid result index");
  }
};

class stablehlo_sine : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 1; }
  unsigned getNumAttributes() const override { return 0; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpOrComplexTensor;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "operand";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpOrComplexTensor;
    }
    assert(false && "Invalid result index");
  }
};

class stablehlo_slice : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 1; }
  unsigned getNumAttributes() const override { return 3; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "operand";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return ::mlir::DenseIntElementsAttr();
      case 1: return ::mlir::DenseIntElementsAttr();
      case 2: return ::mlir::DenseIntElementsAttr();
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
      case 0: return "start_indices";
      case 1: return "limit_indices";
      case 2: return "strides";
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
      case 0: return true;
      case 1: return true;
      case 2: return true;
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    attrs.push_back(attrGen->genDenseIntElementsAttr("start_indices"));
    attrs.push_back(attrGen->genDenseIntElementsAttr("limit_indices"));
    attrs.push_back(attrGen->genDenseIntElementsAttr("strides"));
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class stablehlo_sort : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 1; }
  unsigned getNumAttributes() const override { return 2; }
  unsigned getNumRegions() const override { return 1; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "inputs";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return ::mlir::IntegerAttr();
      case 1: return ::mlir::BoolAttr();
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
      case 0: return "dimension";
      case 1: return "is_stable";
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
      case 0: return false;
      case 1: return false;
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    // attrs.push_back(attrGen->genIntegerAttr("dimension"));
    // attrs.push_back(attrGen->genBoolAttr("is_stable"));
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class stablehlo_sqrt : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 1; }
  unsigned getNumAttributes() const override { return 0; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpOrComplexTensor;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "operand";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpOrComplexTensor;
    }
    assert(false && "Invalid result index");
  }
};

class stablehlo_subtract : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 2; }
  unsigned getNumAttributes() const override { return 0; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
      case 1: return HLO_Tensor;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "lhs";
      case 1: return "rhs";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class stablehlo_tanh : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 1; }
  unsigned getNumAttributes() const override { return 0; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpOrComplexTensor;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "operand";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpOrComplexTensor;
    }
    assert(false && "Invalid result index");
  }
};

class stablehlo_torch_index_select : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 2; }
  unsigned getNumAttributes() const override { return 2; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
      case 1: return HLO_Tensor;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "operand";
      case 1: return "index";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return ::mlir::IntegerAttr();
      case 1: return ::mlir::IntegerAttr();
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
      case 0: return "dim";
      case 1: return "batch_dims";
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
      case 0: return true;
      case 1: return true;
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    attrs.push_back(attrGen->genIntegerAttr("dim"));
    attrs.push_back(attrGen->genIntegerAttr("batch_dims"));
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class stablehlo_trace : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 1; }
  unsigned getNumAttributes() const override { return 1; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 0; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "operand";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return ::mlir::StringAttr();
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
      case 0: return "tag";
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
      case 0: return true;
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    attrs.push_back(attrGen->genStringAttr("tag"));
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid result index");
  }
};

class stablehlo_transpose : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 1; }
  unsigned getNumAttributes() const override { return 1; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "operand";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return ::mlir::DenseIntElementsAttr();
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
      case 0: return "permutation";
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
      case 0: return true;
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    attrs.push_back(attrGen->genDenseIntElementsAttr("permutation"));
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class stablehlo_triangular_solve : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 2; }
  unsigned getNumAttributes() const override { return 4; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpOrComplexTensor;
      case 1: return HLO_FpOrComplexTensor;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "a";
      case 1: return "b";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return ::mlir::BoolAttr();
      case 1: return ::mlir::BoolAttr();
      case 2: return ::mlir::BoolAttr();
      case 3: return ::mlir::stablehlo::TransposeAttr();
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
      case 0: return "left_side";
      case 1: return "lower";
      case 2: return "unit_diagonal";
      case 3: return "transpose_a";
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
      case 0: return true;
      case 1: return true;
      case 2: return true;
      case 3: return true;
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    attrs.push_back(attrGen->genBoolAttr("left_side"));
    attrs.push_back(attrGen->genBoolAttr("lower"));
    attrs.push_back(attrGen->genBoolAttr("unit_diagonal"));
    attrs.push_back(attrGen->genStablehloTransposeAttr("transpose_a"));
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_FpOrComplexTensor;
    }
    assert(false && "Invalid result index");
  }
};

class stablehlo_tuple : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 1; }
  unsigned getNumAttributes() const override { return 0; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_TensorOrTokenOrTuple;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "val";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tuple;
    }
    assert(false && "Invalid result index");
  }
};

class stablehlo_unary_einsum : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 1; }
  unsigned getNumAttributes() const override { return 1; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "operand";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return ::mlir::StringAttr();
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
      case 0: return "einsum_config";
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
      case 0: return true;
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    attrs.push_back(attrGen->genStringAttr("einsum_config"));
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class stablehlo_uniform_dequantize : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 1; }
  unsigned getNumAttributes() const override { return 0; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_QuantizedIntTensor;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "operand";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return anonymous_694;
    }
    assert(false && "Invalid result index");
  }
};

class stablehlo_uniform_quantize : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 1; }
  unsigned getNumAttributes() const override { return 0; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return anonymous_688;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "operand";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_QuantizedIntTensor;
    }
    assert(false && "Invalid result index");
  }
};

class stablehlo_while : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 1; }
  unsigned getNumAttributes() const override { return 0; }
  unsigned getNumRegions() const override { return 2; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_TensorOrToken;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "operand";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_TensorOrToken;
    }
    assert(false && "Invalid result index");
  }
};

class stablehlo_xor : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 2; }
  unsigned getNumAttributes() const override { return 0; }
  unsigned getNumRegions() const override { return 0; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_PredOrIntTensor;
      case 1: return HLO_PredOrIntTensor;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "lhs";
      case 1: return "rhs";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return HLO_Tensor;
    }
    assert(false && "Invalid result index");
  }
};

class linalg_transpose : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 2; }
  unsigned getNumAttributes() const override { return 1; }
  unsigned getNumRegions() const override { return 1; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return TensorOrMemref;
      case 1: return TensorOrMemref;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "input";
      case 1: return "init";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
      case 0: return ::mlir::DenseI64ArrayAttr();
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
      case 0: return "permutation";
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
      case 0: return true;
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    attrs.push_back(attrGen->genDenseI64ArrayAttr("permutation"));
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return AnyTensor;
    }
    assert(false && "Invalid result index");
  }
};

class linalg_vecmat : public GrammarOp {
public:
  unsigned getNumOperands() const override { return 2; }
  unsigned getNumAttributes() const override { return 0; }
  unsigned getNumRegions() const override { return 1; }
  unsigned getNumResults() const override { return 1; }
  OpAndResType getOperandType(unsigned index) const override {
    switch (index) {
      case 0: return AnyType;
      case 1: return AnyShaped;
    }
    assert(false && "Invalid operand index");
  }
  std::string getOperandName(unsigned index) const override {
    switch (index) {
      case 0: return "inputs";
      case 1: return "outputs";
    }
    assert(false && "Invalid operand index");
  }
  mlir::Attribute getAttributeType(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::string getAttributeName(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  bool isAttributeRequired(unsigned index) const override {
    switch (index) {
    }
    assert(false && "Invalid attribute index");
  }
  std::vector<std::vector<mlir::Attribute>> genAttributes(AttributeGeneratorBasePtr attrGen) const override {
    std::vector<std::vector<mlir::Attribute>> attrs;
    return attrs;
  }
  OpAndResType getResultType(unsigned index) const override {
    switch (index) {
      case 0: return AnyRankedTensor;
    }
    assert(false && "Invalid result index");
  }
};

GrammarOpPtr createGrammarOp(std::string name) {
  if (name == "linalg.batch_matmul")
    return std::make_unique<linalg_batch_matmul>();
  if (name == "linalg.batch_matvec")
    return std::make_unique<linalg_batch_matvec>();
  if (name == "linalg.batch_reduce_matmul")
    return std::make_unique<linalg_batch_reduce_matmul>();
  if (name == "chlo.acos")
    return std::make_unique<chlo_acos>();
  if (name == "chlo.acosh")
    return std::make_unique<chlo_acosh>();
  if (name == "chlo.asin")
    return std::make_unique<chlo_asin>();
  if (name == "chlo.asinh")
    return std::make_unique<chlo_asinh>();
  if (name == "chlo.atan")
    return std::make_unique<chlo_atan>();
  if (name == "chlo.atanh")
    return std::make_unique<chlo_atanh>();
  if (name == "chlo.bessel_i1e")
    return std::make_unique<chlo_bessel_i1e>();
  if (name == "chlo.broadcast_add")
    return std::make_unique<chlo_broadcast_add>();
  if (name == "chlo.broadcast_and")
    return std::make_unique<chlo_broadcast_and>();
  if (name == "chlo.broadcast_atan2")
    return std::make_unique<chlo_broadcast_atan2>();
  if (name == "chlo.broadcast_compare")
    return std::make_unique<chlo_broadcast_compare>();
  if (name == "chlo.broadcast_complex")
    return std::make_unique<chlo_broadcast_complex>();
  if (name == "chlo.broadcast_divide")
    return std::make_unique<chlo_broadcast_divide>();
  if (name == "chlo.broadcast_maximum")
    return std::make_unique<chlo_broadcast_maximum>();
  if (name == "chlo.broadcast_minimum")
    return std::make_unique<chlo_broadcast_minimum>();
  if (name == "chlo.broadcast_multiply")
    return std::make_unique<chlo_broadcast_multiply>();
  if (name == "chlo.broadcast_next_after")
    return std::make_unique<chlo_broadcast_next_after>();
  if (name == "chlo.broadcast_or")
    return std::make_unique<chlo_broadcast_or>();
  if (name == "chlo.broadcast_polygamma")
    return std::make_unique<chlo_broadcast_polygamma>();
  if (name == "chlo.broadcast_power")
    return std::make_unique<chlo_broadcast_power>();
  if (name == "chlo.broadcast_remainder")
    return std::make_unique<chlo_broadcast_remainder>();
  if (name == "chlo.broadcast_select")
    return std::make_unique<chlo_broadcast_select>();
  if (name == "chlo.broadcast_shift_left")
    return std::make_unique<chlo_broadcast_shift_left>();
  if (name == "chlo.broadcast_shift_right_arithmetic")
    return std::make_unique<chlo_broadcast_shift_right_arithmetic>();
  if (name == "chlo.broadcast_shift_right_logical")
    return std::make_unique<chlo_broadcast_shift_right_logical>();
  if (name == "chlo.broadcast_subtract")
    return std::make_unique<chlo_broadcast_subtract>();
  if (name == "chlo.broadcast_xor")
    return std::make_unique<chlo_broadcast_xor>();
  if (name == "chlo.broadcast_zeta")
    return std::make_unique<chlo_broadcast_zeta>();
  if (name == "chlo.conj")
    return std::make_unique<chlo_conj>();
  if (name == "chlo.constant_like")
    return std::make_unique<chlo_constant_like>();
  if (name == "chlo.constant")
    return std::make_unique<chlo_constant>();
  if (name == "chlo.cosh")
    return std::make_unique<chlo_cosh>();
  if (name == "chlo.digamma")
    return std::make_unique<chlo_digamma>();
  if (name == "chlo.dynamic_reshape")
    return std::make_unique<chlo_dynamic_reshape>();
  if (name == "chlo.erf")
    return std::make_unique<chlo_erf>();
  if (name == "chlo.erfc")
    return std::make_unique<chlo_erfc>();
  if (name == "chlo.is_inf")
    return std::make_unique<chlo_is_inf>();
  if (name == "chlo.is_neg_inf")
    return std::make_unique<chlo_is_neg_inf>();
  if (name == "chlo.is_pos_inf")
    return std::make_unique<chlo_is_pos_inf>();
  if (name == "chlo.lgamma")
    return std::make_unique<chlo_lgamma>();
  if (name == "chlo.minimum_broadcast_shapes")
    return std::make_unique<chlo_minimum_broadcast_shapes>();
  if (name == "chlo.next_after")
    return std::make_unique<chlo_next_after>();
  if (name == "chlo.polygamma")
    return std::make_unique<chlo_polygamma>();
  if (name == "chlo.rank_specialization_cluster")
    return std::make_unique<chlo_rank_specialization_cluster>();
  if (name == "chlo.rank_specialization_cluster_yield")
    return std::make_unique<chlo_rank_specialization_cluster_yield>();
  if (name == "chlo.sinh")
    return std::make_unique<chlo_sinh>();
  if (name == "chlo.tan")
    return std::make_unique<chlo_tan>();
  if (name == "chlo.top_k")
    return std::make_unique<chlo_top_k>();
  if (name == "chlo.zeta")
    return std::make_unique<chlo_zeta>();
  if (name == "linalg.conv_1d_ncw_fcw")
    return std::make_unique<linalg_conv_1d_ncw_fcw>();
  if (name == "linalg.conv_1d_nwc_wcf")
    return std::make_unique<linalg_conv_1d_nwc_wcf>();
  if (name == "linalg.conv_1d")
    return std::make_unique<linalg_conv_1d>();
  if (name == "linalg.conv_2d_nchw_fchw")
    return std::make_unique<linalg_conv_2d_nchw_fchw>();
  if (name == "linalg.conv_2d_ngchw_fgchw")
    return std::make_unique<linalg_conv_2d_ngchw_fgchw>();
  if (name == "linalg.conv_2d_nhwc_fhwc")
    return std::make_unique<linalg_conv_2d_nhwc_fhwc>();
  if (name == "linalg.conv_2d_nhwc_hwcf")
    return std::make_unique<linalg_conv_2d_nhwc_hwcf>();
  if (name == "linalg.conv_2d_nhwc_hwcf_q")
    return std::make_unique<linalg_conv_2d_nhwc_hwcf_q>();
  if (name == "linalg.conv_2d")
    return std::make_unique<linalg_conv_2d>();
  if (name == "linalg.conv_3d_ndhwc_dhwcf")
    return std::make_unique<linalg_conv_3d_ndhwc_dhwcf>();
  if (name == "linalg.conv_3d")
    return std::make_unique<linalg_conv_3d>();
  if (name == "linalg.copy")
    return std::make_unique<linalg_copy>();
  if (name == "linalg.depthwise_conv_1d_nwc_wc")
    return std::make_unique<linalg_depthwise_conv_1d_nwc_wc>();
  if (name == "linalg.depthwise_conv_1d_nwc_wcm")
    return std::make_unique<linalg_depthwise_conv_1d_nwc_wcm>();
  if (name == "linalg.depthwise_conv_2d_nchw_chw")
    return std::make_unique<linalg_depthwise_conv_2d_nchw_chw>();
  if (name == "linalg.depthwise_conv_2d_nhwc_hwc")
    return std::make_unique<linalg_depthwise_conv_2d_nhwc_hwc>();
  if (name == "linalg.depthwise_conv_2d_nhwc_hwc_q")
    return std::make_unique<linalg_depthwise_conv_2d_nhwc_hwc_q>();
  if (name == "linalg.depthwise_conv_2d_nhwc_hwcm")
    return std::make_unique<linalg_depthwise_conv_2d_nhwc_hwcm>();
  if (name == "linalg.depthwise_conv_2d_nhwc_hwcm_q")
    return std::make_unique<linalg_depthwise_conv_2d_nhwc_hwcm_q>();
  if (name == "linalg.depthwise_conv_3d_ndhwc_dhwc")
    return std::make_unique<linalg_depthwise_conv_3d_ndhwc_dhwc>();
  if (name == "linalg.depthwise_conv_3d_ndhwc_dhwcm")
    return std::make_unique<linalg_depthwise_conv_3d_ndhwc_dhwcm>();
  if (name == "linalg.dot")
    return std::make_unique<linalg_dot>();
  if (name == "linalg.elemwise_binary")
    return std::make_unique<linalg_elemwise_binary>();
  if (name == "linalg.elemwise_unary")
    return std::make_unique<linalg_elemwise_unary>();
  if (name == "linalg.fill")
    return std::make_unique<linalg_fill>();
  if (name == "linalg.fill_rng_2d")
    return std::make_unique<linalg_fill_rng_2d>();
  if (name == "linalg.generic")
    return std::make_unique<linalg_generic>();
  if (name == "linalg.index")
    return std::make_unique<linalg_index>();
  if (name == "linalg.yield")
    return std::make_unique<linalg_yield>();
  if (name == "linalg.map")
    return std::make_unique<linalg_map>();
  if (name == "linalg.matmul")
    return std::make_unique<linalg_matmul>();
  if (name == "linalg.matmul_unsigned")
    return std::make_unique<linalg_matmul_unsigned>();
  if (name == "linalg.matvec")
    return std::make_unique<linalg_matvec>();
  if (name == "linalg.mmt4d")
    return std::make_unique<linalg_mmt4d>();
  if (name == "linalg.pooling_nchw_max")
    return std::make_unique<linalg_pooling_nchw_max>();
  if (name == "linalg.pooling_nchw_sum")
    return std::make_unique<linalg_pooling_nchw_sum>();
  if (name == "linalg.pooling_ndhwc_max")
    return std::make_unique<linalg_pooling_ndhwc_max>();
  if (name == "linalg.pooling_ndhwc_min")
    return std::make_unique<linalg_pooling_ndhwc_min>();
  if (name == "linalg.pooling_ndhwc_sum")
    return std::make_unique<linalg_pooling_ndhwc_sum>();
  if (name == "linalg.pooling_nhwc_max")
    return std::make_unique<linalg_pooling_nhwc_max>();
  if (name == "linalg.pooling_nhwc_max_unsigned")
    return std::make_unique<linalg_pooling_nhwc_max_unsigned>();
  if (name == "linalg.pooling_nhwc_min")
    return std::make_unique<linalg_pooling_nhwc_min>();
  if (name == "linalg.pooling_nhwc_min_unsigned")
    return std::make_unique<linalg_pooling_nhwc_min_unsigned>();
  if (name == "linalg.pooling_nhwc_sum")
    return std::make_unique<linalg_pooling_nhwc_sum>();
  if (name == "linalg.quantized_batch_matmul")
    return std::make_unique<linalg_quantized_batch_matmul>();
  if (name == "linalg.quantized_matmul")
    return std::make_unique<linalg_quantized_matmul>();
  if (name == "linalg.reduce")
    return std::make_unique<linalg_reduce>();
  if (name == "stablehlo.abs")
    return std::make_unique<stablehlo_abs>();
  if (name == "stablehlo.add")
    return std::make_unique<stablehlo_add>();
  if (name == "stablehlo.after_all")
    return std::make_unique<stablehlo_after_all>();
  if (name == "stablehlo.all_gather")
    return std::make_unique<stablehlo_all_gather>();
  if (name == "stablehlo.all_reduce")
    return std::make_unique<stablehlo_all_reduce>();
  if (name == "stablehlo.all_to_all")
    return std::make_unique<stablehlo_all_to_all>();
  if (name == "stablehlo.and")
    return std::make_unique<stablehlo_and>();
  if (name == "stablehlo.atan2")
    return std::make_unique<stablehlo_atan2>();
  if (name == "stablehlo.batch_norm_grad")
    return std::make_unique<stablehlo_batch_norm_grad>();
  if (name == "stablehlo.batch_norm_inference")
    return std::make_unique<stablehlo_batch_norm_inference>();
  if (name == "stablehlo.batch_norm_training")
    return std::make_unique<stablehlo_batch_norm_training>();
  if (name == "stablehlo.bitcast_convert")
    return std::make_unique<stablehlo_bitcast_convert>();
  if (name == "stablehlo.broadcast_in_dim")
    return std::make_unique<stablehlo_broadcast_in_dim>();
  if (name == "stablehlo.broadcast")
    return std::make_unique<stablehlo_broadcast>();
  if (name == "stablehlo.case")
    return std::make_unique<stablehlo_case>();
  if (name == "stablehlo.cbrt")
    return std::make_unique<stablehlo_cbrt>();
  if (name == "stablehlo.ceil")
    return std::make_unique<stablehlo_ceil>();
  if (name == "stablehlo.cholesky")
    return std::make_unique<stablehlo_cholesky>();
  if (name == "stablehlo.clamp")
    return std::make_unique<stablehlo_clamp>();
  if (name == "stablehlo.count_leading_zeros")
    return std::make_unique<stablehlo_count_leading_zeros>();
  if (name == "stablehlo.collective_permute")
    return std::make_unique<stablehlo_collective_permute>();
  if (name == "stablehlo.compare")
    return std::make_unique<stablehlo_compare>();
  if (name == "stablehlo.complex")
    return std::make_unique<stablehlo_complex>();
  if (name == "stablehlo.compute_reshape_shape")
    return std::make_unique<stablehlo_compute_reshape_shape>();
  if (name == "stablehlo.concatenate")
    return std::make_unique<stablehlo_concatenate>();
  if (name == "stablehlo.constant")
    return std::make_unique<stablehlo_constant>();
  if (name == "stablehlo.convert")
    return std::make_unique<stablehlo_convert>();
  if (name == "stablehlo.convolution")
    return std::make_unique<stablehlo_convolution>();
  if (name == "stablehlo.cosine")
    return std::make_unique<stablehlo_cosine>();
  if (name == "stablehlo.create_token")
    return std::make_unique<stablehlo_create_token>();
  if (name == "stablehlo.cross-replica-sum")
    return std::make_unique<stablehlo_cross_replica_sum>();
  if (name == "stablehlo.cstr_reshapable")
    return std::make_unique<stablehlo_cstr_reshapable>();
  if (name == "stablehlo.custom_call")
    return std::make_unique<stablehlo_custom_call>();
  if (name == "stablehlo.divide")
    return std::make_unique<stablehlo_divide>();
  if (name == "stablehlo.dot_general")
    return std::make_unique<stablehlo_dot_general>();
  if (name == "stablehlo.dot")
    return std::make_unique<stablehlo_dot>();
  if (name == "stablehlo.dynamic_broadcast_in_dim")
    return std::make_unique<stablehlo_dynamic_broadcast_in_dim>();
  if (name == "stablehlo.dynamic_conv")
    return std::make_unique<stablehlo_dynamic_conv>();
  if (name == "stablehlo.dynamic_gather")
    return std::make_unique<stablehlo_dynamic_gather>();
  if (name == "stablehlo.dynamic_iota")
    return std::make_unique<stablehlo_dynamic_iota>();
  if (name == "stablehlo.dynamic_pad")
    return std::make_unique<stablehlo_dynamic_pad>();
  if (name == "stablehlo.dynamic_reshape")
    return std::make_unique<stablehlo_dynamic_reshape>();
  if (name == "stablehlo.dynamic_slice")
    return std::make_unique<stablehlo_dynamic_slice>();
  if (name == "stablehlo.dynamic_update_slice")
    return std::make_unique<stablehlo_dynamic_update_slice>();
  if (name == "stablehlo.einsum")
    return std::make_unique<stablehlo_einsum>();
  if (name == "stablehlo.exponential")
    return std::make_unique<stablehlo_exponential>();
  if (name == "stablehlo.exponential_minus_one")
    return std::make_unique<stablehlo_exponential_minus_one>();
  if (name == "stablehlo.fft")
    return std::make_unique<stablehlo_fft>();
  if (name == "stablehlo.floor")
    return std::make_unique<stablehlo_floor>();
  if (name == "stablehlo.gather")
    return std::make_unique<stablehlo_gather>();
  if (name == "stablehlo.get_dimension_size")
    return std::make_unique<stablehlo_get_dimension_size>();
  if (name == "stablehlo.get_tuple_element")
    return std::make_unique<stablehlo_get_tuple_element>();
  if (name == "stablehlo.if")
    return std::make_unique<stablehlo_if>();
  if (name == "stablehlo.imag")
    return std::make_unique<stablehlo_imag>();
  if (name == "stablehlo.infeed")
    return std::make_unique<stablehlo_infeed>();
  if (name == "stablehlo.iota")
    return std::make_unique<stablehlo_iota>();
  if (name == "stablehlo.is_finite")
    return std::make_unique<stablehlo_is_finite>();
  if (name == "stablehlo.log_plus_one")
    return std::make_unique<stablehlo_log_plus_one>();
  if (name == "stablehlo.log")
    return std::make_unique<stablehlo_log>();
  if (name == "stablehlo.logistic")
    return std::make_unique<stablehlo_logistic>();
  if (name == "stablehlo.map")
    return std::make_unique<stablehlo_map>();
  if (name == "stablehlo.maximum")
    return std::make_unique<stablehlo_maximum>();
  if (name == "stablehlo.minimum")
    return std::make_unique<stablehlo_minimum>();
  if (name == "stablehlo.multiply")
    return std::make_unique<stablehlo_multiply>();
  if (name == "stablehlo.negate")
    return std::make_unique<stablehlo_negate>();
  if (name == "stablehlo.not")
    return std::make_unique<stablehlo_not>();
  if (name == "stablehlo.optimization_barrier")
    return std::make_unique<stablehlo_optimization_barrier>();
  if (name == "stablehlo.or")
    return std::make_unique<stablehlo_or>();
  if (name == "stablehlo.outfeed")
    return std::make_unique<stablehlo_outfeed>();
  if (name == "stablehlo.pad")
    return std::make_unique<stablehlo_pad>();
  if (name == "stablehlo.popcnt")
    return std::make_unique<stablehlo_popcnt>();
  if (name == "stablehlo.power")
    return std::make_unique<stablehlo_power>();
  if (name == "stablehlo.real_dynamic_slice")
    return std::make_unique<stablehlo_real_dynamic_slice>();
  if (name == "stablehlo.real")
    return std::make_unique<stablehlo_real>();
  if (name == "stablehlo.recv")
    return std::make_unique<stablehlo_recv>();
  if (name == "stablehlo.reduce")
    return std::make_unique<stablehlo_reduce>();
  if (name == "stablehlo.reduce_precision")
    return std::make_unique<stablehlo_reduce_precision>();
  if (name == "stablehlo.reduce_scatter")
    return std::make_unique<stablehlo_reduce_scatter>();
  if (name == "stablehlo.reduce_window")
    return std::make_unique<stablehlo_reduce_window>();
  if (name == "stablehlo.remainder")
    return std::make_unique<stablehlo_remainder>();
  if (name == "stablehlo.replica_id")
    return std::make_unique<stablehlo_replica_id>();
  if (name == "stablehlo.reshape")
    return std::make_unique<stablehlo_reshape>();
  if (name == "stablehlo.return")
    return std::make_unique<stablehlo_return>();
  if (name == "stablehlo.reverse")
    return std::make_unique<stablehlo_reverse>();
  if (name == "stablehlo.rng_bit_generator")
    return std::make_unique<stablehlo_rng_bit_generator>();
  if (name == "stablehlo.rng")
    return std::make_unique<stablehlo_rng>();
  if (name == "stablehlo.round_nearest_even")
    return std::make_unique<stablehlo_round_nearest_even>();
  if (name == "stablehlo.round_nearest_afz")
    return std::make_unique<stablehlo_round_nearest_afz>();
  if (name == "stablehlo.rsqrt")
    return std::make_unique<stablehlo_rsqrt>();
  if (name == "stablehlo.scatter")
    return std::make_unique<stablehlo_scatter>();
  if (name == "stablehlo.select_and_scatter")
    return std::make_unique<stablehlo_select_and_scatter>();
  if (name == "stablehlo.select")
    return std::make_unique<stablehlo_select>();
  if (name == "stablehlo.send")
    return std::make_unique<stablehlo_send>();
  if (name == "stablehlo.set_dimension_size")
    return std::make_unique<stablehlo_set_dimension_size>();
  if (name == "stablehlo.shift_left")
    return std::make_unique<stablehlo_shift_left>();
  if (name == "stablehlo.shift_right_arithmetic")
    return std::make_unique<stablehlo_shift_right_arithmetic>();
  if (name == "stablehlo.shift_right_logical")
    return std::make_unique<stablehlo_shift_right_logical>();
  if (name == "stablehlo.sign")
    return std::make_unique<stablehlo_sign>();
  if (name == "stablehlo.sine")
    return std::make_unique<stablehlo_sine>();
  if (name == "stablehlo.slice")
    return std::make_unique<stablehlo_slice>();
  if (name == "stablehlo.sort")
    return std::make_unique<stablehlo_sort>();
  if (name == "stablehlo.sqrt")
    return std::make_unique<stablehlo_sqrt>();
  if (name == "stablehlo.subtract")
    return std::make_unique<stablehlo_subtract>();
  if (name == "stablehlo.tanh")
    return std::make_unique<stablehlo_tanh>();
  if (name == "stablehlo.torch_index_select")
    return std::make_unique<stablehlo_torch_index_select>();
  if (name == "stablehlo.trace")
    return std::make_unique<stablehlo_trace>();
  if (name == "stablehlo.transpose")
    return std::make_unique<stablehlo_transpose>();
  if (name == "stablehlo.triangular_solve")
    return std::make_unique<stablehlo_triangular_solve>();
  if (name == "stablehlo.tuple")
    return std::make_unique<stablehlo_tuple>();
  if (name == "stablehlo.unary_einsum")
    return std::make_unique<stablehlo_unary_einsum>();
  if (name == "stablehlo.uniform_dequantize")
    return std::make_unique<stablehlo_uniform_dequantize>();
  if (name == "stablehlo.uniform_quantize")
    return std::make_unique<stablehlo_uniform_quantize>();
  if (name == "stablehlo.while")
    return std::make_unique<stablehlo_while>();
  if (name == "stablehlo.xor")
    return std::make_unique<stablehlo_xor>();
  if (name == "linalg.transpose")
    return std::make_unique<linalg_transpose>();
  if (name == "linalg.vecmat")
    return std::make_unique<linalg_vecmat>();
  assert(false && "Invalid op name");
}

} // namespace grammar
