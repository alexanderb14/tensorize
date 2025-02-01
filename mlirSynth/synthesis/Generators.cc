#include "Generators.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/Location.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "synthesis/Grammar.h"
#include "llvm/Support/raw_ostream.h"

#include <random>
#include <set>

using namespace mlir;

// Utility functions
// -----------------------------------------------------------------------------
int randomInteger(int min, int max) {
  static std::random_device rd;
  static std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(min, max);
  return dis(gen);
}

void printAttributes(
    std::vector<std::pair<Attribute, grammar::OpAndResType>> &attributes) {
  llvm::outs() << "Attributes:"
               << "\n--------\n";
  for (auto attr : attributes) {
    attr.first.dump();
    llvm::outs() << opAndResTypeToString(attr.second) << "\n";
    llvm::outs() << "---------\n";
  }
}

// Attribute generators
// -----------------------------------------------------------------------------
std::vector<::llvm::SmallVector<int64_t>>
genShapes(std::vector<mlir::Type> &argTypes) {
  std::vector<::llvm::SmallVector<int64_t>> ret;

  for (auto argType : argTypes) {
    if (!argType.isa<ShapedType>())
      continue;

    auto shape = argType.cast<ShapedType>().getShape();

    // Same shape as the argument: E.g. [3, 5, 7] -> [3, 5, 7]
    ret.emplace_back(shape.begin(), shape.end());

    // Dimension at previous last inserted: E.g. [3, 5, 7] -> [3, 5, 1, 7]
    ::llvm::SmallVector<int64_t> shapeWithOne(shape.begin(), shape.end());
    shapeWithOne.insert(shapeWithOne.end() - 1, 1);
    ret.emplace_back(shapeWithOne);

    // Leading dimension inserted: E.g. [5] -> [1, 5] or [3, 5] -> [1, 3, 5]
    ::llvm::SmallVector<int64_t> shapeWithOneLeading(shape.begin(),
                                                     shape.end());
    shapeWithOneLeading.insert(shapeWithOneLeading.begin(), 1);
    ret.emplace_back(shapeWithOneLeading);

    // Trailing dimension inserted: E.g. [5] -> [5, 1] or [3, 5] -> [3, 5, 1]
    ::llvm::SmallVector<int64_t> shapeWithOneTrailing(shape.begin(),
                                                      shape.end());
    shapeWithOneTrailing.insert(shapeWithOneTrailing.end(), 1);
    ret.emplace_back(shapeWithOneTrailing);

    // Transpose: E.g. [3, 5] -> [5, 3]
    ::llvm::SmallVector<int64_t> shapeTransposed(shape.begin(), shape.end());
    std::reverse(shapeTransposed.begin(), shapeTransposed.end());
    ret.emplace_back(shapeTransposed);

    // Reverse: E.g. when rank 1, [1, 0] or rank 2, [2, 1, 0]
    llvm::SmallVector<int64_t> sequenceReverse;
    for (int i = shape.size() - 1; i >= 0; i--) {
      sequenceReverse.push_back(i);
    }
    ret.emplace_back(sequenceReverse);

    // All permutations
    for (unsigned i = 0; i <= 5; i++) {
      for (unsigned j = 0; j <= 5; j++) {
        ::llvm::SmallVector<int64_t> newShape(2, 0);
        newShape[0] = i;
        newShape[1] = j;
        ret.emplace_back(newShape);
      }
    }
  }

  // Remove duplicates
  std::sort(ret.begin(), ret.end());
  ret.erase(std::unique(ret.begin(), ret.end()), ret.end());

  // Filter all 1 element shapes
  ret.erase(std::remove_if(ret.begin(), ret.end(),
                           [](const ::llvm::SmallVector<int64_t> &shape) {
                             return shape.size() == 1;
                           }),
            ret.end());

  return ret;
}

std::vector<::llvm::SmallVector<int64_t>>
genArgShapes(std::vector<mlir::Type> &argTypes) {
  std::vector<::llvm::SmallVector<int64_t>> ret;

  for (auto argType : argTypes) {
    if (!argType.isa<ShapedType>())
      continue;

    auto shape = argType.cast<ShapedType>().getShape();

    // Same shape as the argument: E.g. [3, 5, 7] -> [3, 5, 7]
    ret.emplace_back(shape.begin(), shape.end());
  }

  return ret;
}

std::vector<std::pair<Attribute, grammar::OpAndResType>>
genShapeAttributes(OpBuilder &builder, std::vector<mlir::Type> &argTypes,
                   mlir::Type &returnType) {
  std::vector<std::pair<Attribute, grammar::OpAndResType>> attributes;

  auto shapes = genShapes(argTypes);
  for (auto &shape : shapes) {
    attributes.emplace_back(builder.getI64TensorAttr(shape),
                            grammar::OpAndResType::HLO_DimensionTensor);
  }

  return attributes;
}

std::vector<int64_t> genUnaries(std::vector<mlir::Type> &argTypes,
                                mlir::Type &returnType) {
  std::set<int64_t> scalars;

  // Add 0 and 1.
  scalars.insert(0);
  scalars.insert(1);

  for (auto argType : argTypes) {
    if (!argType.isa<ShapedType>())
      continue;
    auto shape = argType.cast<ShapedType>().getShape();

    for (unsigned i = 0; i < shape.size(); i++) {
      scalars.insert(i);
    }
  }

  return std::vector<int64_t>(scalars.begin(), scalars.end());
}

std::vector<std::pair<Attribute, grammar::OpAndResType>>
genUnaryAttributes(OpBuilder &builder, std::vector<mlir::Type> &argTypes,
                   mlir::Type &returnType) {
  std::vector<std::pair<Attribute, grammar::OpAndResType>> attributes;

  auto unaries = genUnaries(argTypes, returnType);
  // Float
  for (auto &unary : unaries) {
    Attribute unaryAttr = builder.getF64FloatAttr((double)unary);
    Type type = RankedTensorType::get({}, builder.getF64Type());
    auto attrDense = DenseElementsAttr::get(type, {unaryAttr});

    attributes.emplace_back(attrDense, grammar::OpAndResType::HLO_Tensor);
  }
  // Int
  for (auto &unary : unaries) {
    Attribute unaryAttr = builder.getI64IntegerAttr((double)unary);
    Type type = RankedTensorType::get({}, builder.getI64Type());
    auto attrDense = DenseElementsAttr::get(type, {unaryAttr});

    attributes.emplace_back(attrDense, grammar::OpAndResType::HLO_Tensor);
  }

  return attributes;
}

std::vector<std::pair<Attribute, grammar::OpAndResType>>
genMaskAttributes(OpBuilder &builder, std::vector<mlir::Type> &argTypes,
                  mlir::Type &returnType) {
  std::vector<std::pair<Attribute, grammar::OpAndResType>> attributes;

  auto targetShape = returnType.cast<ShapedType>().getShape();
  if (targetShape.size() == 2) {
    // Create a mask mask with the lower triangle set to 1 and the rest to 0.
    std::vector<Attribute> attrVect;
    for (int i = 0; i < targetShape[0]; i++) {
      for (int j = 0; j < targetShape[1]; j++) {
        if (i >= j) {
          attrVect.push_back(builder.getBoolAttr(true));
        } else {
          attrVect.push_back(builder.getBoolAttr(false));
        }
      }
    }
    Type type = RankedTensorType::get({targetShape[0], targetShape[1]},
                                      builder.getI1Type());
    auto attrDense = DenseElementsAttr::get(type.cast<TensorType>(), attrVect);
    attributes.emplace_back(attrDense, grammar::OpAndResType::HLO_PredTensor);

    // Create a mask where the diagonal is 1 and the rest is 0.
    std::vector<Attribute> attrVect2;
    for (int i = 0; i < targetShape[0]; i++) {
      for (int j = 0; j < targetShape[1]; j++) {
        if (i == j) {
          attrVect2.push_back(builder.getBoolAttr(true));
        } else {
          attrVect2.push_back(builder.getBoolAttr(false));
        }
      }
    }
    type = RankedTensorType::get({targetShape[0], targetShape[1]},
                                 builder.getI1Type());
    attrDense = DenseElementsAttr::get(type.cast<TensorType>(), attrVect2);
    attributes.emplace_back(attrDense, grammar::OpAndResType::HLO_PredTensor);

    // Create a matrix with 0 and one with 1 values.
    for (int value = 0; value < 2; value++) {
      attrVect = std::vector<Attribute>();
      for (int i = 0; i < targetShape[0] * targetShape[1]; i++) {
        attrVect.push_back(builder.getF64FloatAttr(value));
      }
      type = RankedTensorType::get({targetShape[0], targetShape[1]},
                                   builder.getF64Type());
      attrDense = DenseElementsAttr::get(type.cast<TensorType>(), attrVect);
      attributes.emplace_back(attrDense, grammar::OpAndResType::HLO_Tensor);
    }
  }

  return attributes;
}

std::vector<std::pair<Attribute, grammar::OpAndResType>>
genAttributes(MLIRContext &ctx, std::vector<mlir::Type> &argTypes,
              mlir::Type &returnType) {
  std::vector<std::pair<Attribute, grammar::OpAndResType>> attributes;
  OpBuilder builder(&ctx);

  auto shapeAttributes = genShapeAttributes(builder, argTypes, returnType);
  attributes.insert(attributes.end(), shapeAttributes.begin(),
                    shapeAttributes.end());

  auto maskAttributes = genMaskAttributes(builder, argTypes, returnType);
  attributes.insert(attributes.end(), maskAttributes.begin(),
                    maskAttributes.end());

  auto unaryAttributes = genUnaryAttributes(builder, argTypes, returnType);
  attributes.insert(attributes.end(), unaryAttributes.begin(),
                    unaryAttributes.end());

  // printAttributes(attributes);

  return attributes;
}

std::vector<std::pair<Attribute, grammar::OpAndResType>>
genDenseInShapeAttributes(MLIRContext &ctx, std::vector<mlir::Type> types) {
  std::vector<std::pair<Attribute, grammar::OpAndResType>> attributes;
  OpBuilder builder(&ctx);

  SmallVector<double> values = {0.0, 0.1, 1.0, 2.0};

  for (auto i : values) {
    for (auto type : types) {
      if (!type.isa<ShapedType>())
        continue;

      // Create new TensorType with the same shape and element type as the
      // argument.
      auto shape = type.cast<ShapedType>().getShape();
      auto tensorType = RankedTensorType::get(shape, builder.getF64Type());

      auto val = builder.getF64FloatAttr((double)i);
      auto attr = DenseElementsAttr::get(tensorType, val);
      attributes.emplace_back(attr, grammar::OpAndResType::HLO_Tensor);
    }
  }

  return attributes;
}

std::vector<Attribute> genOnesAttributes(MLIRContext &ctx,
                                         std::vector<mlir::Type> &argTypes,
                                         mlir::Type &returnType) {
  int maxDim = 0;

  // Max dimension of all arguments.
  for (auto argType : argTypes) {
    if (!argType.isa<ShapedType>())
      continue;

    auto shape = argType.cast<ShapedType>().getShape();
    maxDim = std::max(maxDim, (int)shape.size());
  }

  // Max dimension of the target shape.
  auto targetShape = returnType.cast<ShapedType>().getShape();
  maxDim = std::max(maxDim, (int)targetShape.size());

  // Create
  OpBuilder builder(&ctx);
  std::vector<Attribute> attributes;

  SmallVector<int64_t> shape;
  for (int i = 0; i < maxDim; i++) {
    shape.push_back(1);
    attributes.push_back(builder.getI64TensorAttr(shape));
  }
  return attributes;
}

std::vector<mlir::Attribute>
AttributeGenerator::genDenseIntElementsAttr(const std::string attributeName) {
  std::vector<mlir::Attribute> attributes;

  OpBuilder builder(&ctx);

  if (attributeName == "strides")
    return genOnesAttributes(ctx, argTypes, returnType);

  auto shapeAttributes = genAttributes(ctx, argTypes, returnType);
  for (auto attr : shapeAttributes) {
    attributes.push_back(attr.first);
  }

  return attributes;
}

std::vector<::llvm::SmallVector<int64_t>>
AttributeGenerator::genLlvmSmallVectorint64t(const std::string attributeName) {
  std::vector<::llvm::SmallVector<int64_t>> attributes;

  attributes.emplace_back();

  auto unaries = genUnaries(argTypes, returnType);
  for (auto &unary : unaries) {
    ::llvm::SmallVector<int64_t> unaryAttr = {unary};
    attributes.push_back(unaryAttr);
  }

  return attributes;
}

// Region generators
// -----------------------------------------------------------------------------
std::vector<std::shared_ptr<Region>> genRegions(MLIRContext &ctx) {
  OpBuilder builder(&ctx);
  auto unknownLoc = UnknownLoc::get(&ctx);

  // Create region with a single block.
  std::shared_ptr<Region> region = std::make_shared<Region>();
  Block *block = new Block();
  region->push_back(block);

  // Add two arguments to the block.
  auto tensorType = RankedTensorType::get({}, builder.getF64Type());
  block->addArgument(tensorType, unknownLoc);
  block->addArgument(tensorType, unknownLoc);

  // Create a add operation with the two arguments.
  auto addOp = builder.create<stablehlo::AddOp>(
      unknownLoc, block->getArgument(0), block->getArgument(1));
  block->push_back(addOp);

  // Create a stablehlo return operation with the result of the add operation.
  auto returnOp = builder.create<stablehlo::ReturnOp>(
      unknownLoc, block->back().getResults());
  block->push_back(returnOp);

  std::vector<std::shared_ptr<Region>> regions;
  regions.push_back(region);

  return regions;
}

// Initial candidate generators
// -----------------------------------------------------------------------------
std::vector<CandidatePtr>
HLOInitialCandidateGenerator::gen(std::vector<mlir::Type> &argTypes,
                                  mlir::Type &returnType) {
  std::vector<CandidatePtr> candidates;

  OpBuilder builder(&ctx);

  // Constant candidates.
  std::vector<std::pair<Attribute, grammar::OpAndResType>> constantAttributes;

  // - Add attributes
  auto attrs = genAttributes(ctx, argTypes, returnType);
  constantAttributes.insert(constantAttributes.end(), attrs.begin(),
                            attrs.end());

  // - Add 0 and 1 in the argument and result types shapes.
  auto zerosAndOnes = genDenseInShapeAttributes(ctx, argTypes);
  constantAttributes.insert(constantAttributes.end(), zerosAndOnes.begin(),
                            zerosAndOnes.end());

  for (auto &attributePair : constantAttributes) {
    auto &attr = attributePair.first;
    auto &type = attributePair.second;

    CandidatePtr candidate(new Candidate({}, type));
    candidate->addOperation(
        ctx, builder.create<stablehlo::ConstantOp>(UnknownLoc::get(&ctx), attr),
        false);
    candidates.push_back(candidate);
  }

  // Argument candidates.
  std::vector<std::pair<mlir::Type, grammar::OpAndResType>> inputs;
  for (auto argType : argTypes) {
    if (argType.isa<ShapedType>()) {
      auto shapedType = argType.cast<ShapedType>();
      inputs.emplace_back(RankedTensorType::get(shapedType.getShape(),
                                                shapedType.getElementType()),
                          grammar::OpAndResType::HLO_Tensor);
      inputs.emplace_back(
          RankedTensorType::get(shapedType.getShape(), builder.getI1Type()),
          grammar::OpAndResType::HLO_PredTensor);
    } else if (argType.isa<FloatType>()) {
      inputs.emplace_back(RankedTensorType::get({}, argType),
                          grammar::OpAndResType::HLO_Tensor);
    } else {
      llvm::outs() << "Type: " << argType << "\n";
      assert(false && "Unsupported type");
    }
  }

  unsigned argId = 0;
  for (auto &input : inputs) {
    CandidatePtr candidate(new Candidate({}, input.second));
    candidate->addArgument(ctx, input.first, argId++);
    candidates.push_back(candidate);
  }

  return candidates;
}

CandidatePtr createTensorInit(mlir::MLIRContext &ctx,
                              llvm::ArrayRef<int64_t> shape, float value) {
  OpBuilder builder(&ctx);
  auto type = grammar::OpAndResType::AnyType;
  CandidatePtr candidate(new Candidate({}, type));

  auto op1 = builder.create<tensor::EmptyOp>(UnknownLoc::get(&ctx), shape,
                                             FloatType::getF64(&ctx));
  candidate->addOperation(ctx, op1, false);

  auto op2 = builder.create<arith::ConstantOp>(
      UnknownLoc::get(&ctx), FloatAttr::get(FloatType::getF64(&ctx), value));
  candidate->addOperation(ctx, op2, false);

  auto op3 = builder.create<linalg::FillOp>(UnknownLoc::get(&ctx),
                                            op2.getResult(), op1.getResult());
  candidate->addOperation(ctx, op3, false);

  return candidate;
}

std::vector<CandidatePtr>
LinalgInitialCandidateGenerator::gen(std::vector<mlir::Type> &argTypes,
                                     mlir::Type &returnType) {
  // Constant candidates.
  std::vector<CandidatePtr> candidates;
  auto shapes = genArgShapes(argTypes);
  for (auto &shape : shapes) {
    candidates.push_back(createTensorInit(ctx, shape, 0.0));
    candidates.push_back(createTensorInit(ctx, shape, 1.0));
  }

  // Argument candidates.
  std::vector<mlir::Type> inputs;
  for (auto argType : argTypes) {
    if (argType.isa<ShapedType>()) {
      auto shapedType = argType.cast<ShapedType>();
      inputs.push_back(RankedTensorType::get(shapedType.getShape(),
                                             shapedType.getElementType()));
    } else if (argType.isa<FloatType>()) {
      inputs.push_back(RankedTensorType::get({}, argType));
    } else {
      llvm::outs() << "Type: " << argType << "\n";
      assert(false && "Unsupported type");
    }
  }

  unsigned argId = 0;
  for (auto &input : inputs) {
    CandidatePtr candidate(new Candidate({}, grammar::OpAndResType::AnyType));
    candidate->addArgument(ctx, input, argId++);
    candidates.push_back(candidate);
  }

  return candidates;
}

// Type aliases
// -----------------------------------------------------------------------------
grammar::OpAndResType getTypeAlias(grammar::OpAndResType type) {
  if (type == grammar::OpAndResType::HLO_FpOrComplexTensor)
    return grammar::OpAndResType::HLO_Tensor;
  return type;
}