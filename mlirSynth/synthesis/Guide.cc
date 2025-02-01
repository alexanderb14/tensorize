#include "Guide.h"

#include "analysis/PolyhedralAnalysis.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/ArrayRef.h"
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/connected_components.hpp>
#include <boost/graph/dijkstra_shortest_paths.hpp>
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/tiernan_all_cycles.hpp>

#include <unordered_map>

using namespace mlir;

template <typename T>
int countNumOps(Operation *op) {
  int numOps = 0;
  op->walk([&](T op) { numOps++; });
  return numOps;
}

int countNumCompareOps(Operation *op, arith::CmpFPredicate predicate) {
  int numOps = 0;
  op->walk([&](arith::CmpFOp op) {
    if (op.getPredicate() == predicate) {
      numOps++;
    }
  });
  return numOps;
}

int computeNumCyclesWithSelfEdges(BoostGraph &g) {
  // Check for cycles
  int numCycles = computeNumCycles(g);

  // Check for self edges
  for (auto v : boost::make_iterator_range(boost::vertices(g))) {
    for (auto e : boost::make_iterator_range(boost::out_edges(v, g))) {
      if (boost::target(e, g) == v) {
        numCycles++;
      }
    }
  }

  return numCycles;
}

llvm::SmallVector<llvm::ArrayRef<int64_t>> getUsedMemrefShapes(Operation *op) {
  llvm::SmallVector<llvm::ArrayRef<int64_t>> memrefShapes;

  // Collect shapes
  for (auto operand : op->getOperands()) {
    // Case 1: Operand is an argument
    if (auto lhsArg = operand.dyn_cast<BlockArgument>()) {
      if (auto memrefType = lhsArg.getType().dyn_cast<MemRefType>()) {
        auto memrefShape = memrefType.getShape();
        memrefShapes.push_back(memrefShape);
      }
    }

    // Case 2: Operand is an affine.load
    if (auto *lhsOp = operand.getDefiningOp()) {
      if (auto loadOp = dyn_cast<AffineLoadOp>(lhsOp)) {
        auto memrefType = loadOp.getMemRefType();
        auto memrefShape = memrefType.getShape();
        memrefShapes.push_back(memrefShape);
      }
    }
  }

  // Recursive walk the use chain
  for (auto operand : op->getOperands()) {
    if (auto *definingOp = operand.getDefiningOp()) {
      auto memrefShapesOperand = getUsedMemrefShapes(definingOp);
      memrefShapes.append(memrefShapesOperand.begin(),
                          memrefShapesOperand.end());
    }
  }

  return memrefShapes;
}

int getMaxArgDim(Operation *op) {
  auto funcOp = dyn_cast<func::FuncOp>(op);
  auto funcType = funcOp.getFunctionType();
  auto funcArgs = funcType.getInputs();

  // Get the maximum dimension of the arguments
  int maxArgDim = 0;
  for (auto arg : funcArgs) {
    if (auto memrefType = arg.dyn_cast<MemRefType>()) {
      auto memrefShape = memrefType.getShape();
      maxArgDim = std::max(maxArgDim, (int)memrefShape.size());
    }
  }

  return maxArgDim;
}

int countNumMultipliedMismatchingMemrefAccesses(Operation *op) {
  int numMultipliedMismatchingMemrefs = 0;

  // Walk over affine.store operations
  llvm::SmallVector<llvm::ArrayRef<int64_t>> allMemrefShapes;
  op->walk([&](arith::MulFOp mulOp) {
    auto memrefShapes = getUsedMemrefShapes(mulOp);
    for (auto memrefShape : memrefShapes) {
      if (memrefShape.empty())
        continue;
      allMemrefShapes.push_back(memrefShape);
    }
  });

  // Check if all memref shapes have a matching dimension with at least one
  // other memref shape. Matching dimensions can be their first-last or
  // last-first dimensions.
  for (auto memrefShape : allMemrefShapes) {
    bool hasMatchingDimension = false;
    for (auto memrefShapeOther : allMemrefShapes) {
      if (memrefShape.size() != memrefShapeOther.size())
        continue;
      if (memrefShape.empty() || memrefShapeOther.empty())
        continue;

      if (memrefShape[0] == memrefShapeOther.back() ||
          memrefShape.back() == memrefShapeOther[0]) {
        hasMatchingDimension = true;
        break;
      }
    }
    if (!hasMatchingDimension) {
      numMultipliedMismatchingMemrefs++;
    }
  }

  return numMultipliedMismatchingMemrefs;
}

int countNumLoadedDimSmallerStoredRank(Operation *op) {
  int num = 0;

  op->walk([&](AffineStoreOp storeOp) {
    auto ty = storeOp.getMemref().getType().cast<MemRefType>();
    int storeRank = ty.getRank();

    // Traverse along the use chain of the store operation.
    std::vector<Value> queue = {storeOp.getValueToStore()};

    while (!queue.empty()) {
      auto value = queue.back();
      queue.pop_back();

      if (isa<BlockArgument>(value))
        continue;

      for (Value operand : value.getDefiningOp()->getOperands()) {
        // Value could be result from operation, or a block argument. Skip block
        // arguments.
        if (isa<BlockArgument>(operand))
          continue;

        if (auto loadOp = dyn_cast<AffineLoadOp>(operand.getDefiningOp())) {
          auto ty = loadOp.getMemRefType();
          int loadRank = ty.getRank();
          if (loadRank < storeRank && loadRank > 0) {
            num++;
          }
        }

        queue.push_back(operand);
      }
    }
  });

  return num;
}

int countNumLoopBoundMaps(Operation *op) {
  int numLoopBoundMaps = 0;
  op->walk([&](AffineForOp forOp) {
    if (!forOp.hasConstantUpperBound() || !forOp.hasConstantLowerBound()) {
      numLoopBoundMaps++;
    }
  });
  return numLoopBoundMaps;
}

bool hasMulsWithSameOperands(Operation *op) {
  bool ret = false;
  op->walk([&](arith::MulFOp mulOp) {
    auto lhs = mulOp.getLhs();
    auto rhs = mulOp.getRhs();
    if (lhs == rhs) {
      ret = true;
    }
  });
  return ret;
}

bool hasDirectMemrefMuls(Operation *op) {
  bool ret = false;
  op->walk([&](arith::MulFOp mulOp) {
    if (auto *lhsOp = mulOp.getLhs().getDefiningOp()) {
      if (auto *rhsOp = mulOp.getRhs().getDefiningOp()) {
        if (auto lhsLoadOp = dyn_cast<AffineLoadOp>(lhsOp)) {
          if (auto rhsLoadOp = dyn_cast<AffineLoadOp>(rhsOp)) {
            ret = true;
          }
        }
      }
    }
  });
  return ret;
}

bool hasContractionPattern(Operation *op) {
  bool ret = false;
  op->walk([&](arith::MulFOp mulOp) {
    if (auto *lhsOp = mulOp.getLhs().getDefiningOp()) {
      if (auto *rhsOp = mulOp.getRhs().getDefiningOp()) {
        if (auto lhsLoadOp = dyn_cast<AffineLoadOp>(lhsOp)) {
          if (auto rhsLoadOp = dyn_cast<AffineLoadOp>(rhsOp)) {
            // Check if the memrefs have the contraction pattern: The last index
            // of the first memref is the same as the first index of the second
            // memref.
            auto lhsIdx =
                lhsLoadOp.getIndices()[lhsLoadOp.getIndices().size() - 1];
            auto rhsIdx = rhsLoadOp.getIndices()[0];

            if (lhsIdx == rhsIdx)
              ret = true;
          }
        }
      }
    }
  });
  return ret;
}

std::vector<std::string> predictOps(std::vector<std::string> &supportedOps,
                                    Operation *op) {
  Scop scop(op);
  auto dg = scop.getDependenceGraph();
  auto g = constructBoostGraph(dg);

  // Element wise heuristics
  std::vector<std::string> ops;
  if (countNumOps<arith::MulFOp>(op) > 0) {
    ops.emplace_back("chlo.broadcast_multiply");
    ops.emplace_back("jnp.multiply");
  }
  if (countNumOps<arith::DivFOp>(op) > 0) {
    ops.emplace_back("chlo.broadcast_divide");
    ops.emplace_back("jnp.divide");
  }
  if (countNumOps<arith::AddFOp>(op) > 0) {
    ops.emplace_back("chlo.broadcast_add");
    ops.emplace_back("jnp.add");
  }
  if (countNumOps<arith::SubFOp>(op) > 0) {
    ops.emplace_back("chlo.broadcast_subtract");
    ops.emplace_back("jnp.subtract");
  }

  if (countNumOps<arith::RemFOp>(op) > 0) {
    ops.emplace_back("jnp.remainder");
  }

  if (countNumOps<math::SqrtOp>(op) > 0) {
    ops.emplace_back("stablehlo.sqrt");
    ops.emplace_back("jnp.sqrt");
  }
  if (countNumOps<arith::SelectOp>(op) > 0) {
    ops.emplace_back("stablehlo.select");
    ops.emplace_back("jnp.where");
  }
  if (countNumOps<arith::CmpFOp>(op) > 0) {
    ops.emplace_back("stablehlo.compare");

    if (countNumCompareOps(op, arith::CmpFPredicate::OEQ) > 0) {
      ops.emplace_back("jnp.equal");
    } else if (countNumCompareOps(op, arith::CmpFPredicate::OGT) > 0) {
      ops.emplace_back("jnp.greater");
    } else if (countNumCompareOps(op, arith::CmpFPredicate::OLT) > 0) {
      ops.emplace_back("jnp.less");
    } else if (countNumCompareOps(op, arith::CmpFPredicate::OGE) > 0) {
      ops.emplace_back("jnp.greater_equal");
    } else if (countNumCompareOps(op, arith::CmpFPredicate::OLE) > 0) {
      ops.emplace_back("jnp.less_equal");
    }
  }

  if (countNumOps<arith::MaxFOp>(op) > 0) {
    ops.emplace_back("jnp.max");
  }

  if (countNumOps<math::ExpOp>(op) > 0) {
    ops.emplace_back("jnp.exp");
  }

  if (hasMulsWithSameOperands(op)) {
    ops.emplace_back("chlo.broadcast_power");
    ops.emplace_back("jnp.power");
  }

  // Transpose heuristics
  if (countNumMultipliedMismatchingMemrefAccesses(op) > 0 ||
      countNumLoadedDimSmallerStoredRank(op) > 0) {
    ops.emplace_back("stablehlo.transpose");
    ops.emplace_back("jnp.transpose");
  }

  // Select heuristics
  if (countNumLoopBoundMaps(op) > 0) {
    ops.emplace_back("stablehlo.select");
    ops.emplace_back("jnp.where");
  }

  // Reduction heuristics
  if (computeNumCyclesWithSelfEdges(g) > 0) {
    if (getMaxArgDim(op) > 2 ||
        countNumMultipliedMismatchingMemrefAccesses(op) > 0 ||
        !hasContractionPattern(op)) {
      ops.emplace_back("stablehlo.dot_general");
      ops.emplace_back("jnp.tensordot");
    } else {
      ops.emplace_back("stablehlo.dot");
      ops.emplace_back("jnp.dot");
    }

    ops.emplace_back("stablehlo.reduce");
    ops.emplace_back("jnp.sum");

    ops.emplace_back("linalg.matmul");
    ops.emplace_back("linalg.matvec");
  }

  // If we didn't match any ops, add all of them.
  if (ops.empty())
    return supportedOps;

  return ops;
}

std::vector<float> predictConstants(mlir::Operation *op) {
  std::vector<float> constants;
  op->walk([&](arith::ConstantOp constantOp) {
    auto attr = constantOp.getValue().cast<FloatAttr>();
    constants.push_back(attr.getValueAsDouble());
    llvm::outs() << "Constant: " << attr.getValueAsDouble() << "\n";
  });

  // Make unique
  std::sort(constants.begin(), constants.end());
  constants.erase(std::unique(constants.begin(), constants.end()),
                  constants.end());

  return constants;
}