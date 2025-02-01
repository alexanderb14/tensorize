//===- TranslateToCpp.cpp - Translating to C++ calls ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "translation/TranslateToPython.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/IndentedOstream.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include <utility>

#define DEBUG_TYPE "translate-to-python"

using namespace mlir;

/// Convenience functions to produce interleaved output with functions returning
/// a LogicalResult. This is different than those in STLExtras as functions used
/// on each element doesn't return a string.
template <typename ForwardIterator, typename UnaryFunctor,
          typename NullaryFunctor>
inline LogicalResult
interleaveWithError(ForwardIterator begin, ForwardIterator end,
                    UnaryFunctor eachFn, NullaryFunctor betweenFn) {
  if (begin == end)
    return success();
  if (failed(eachFn(*begin)))
    return failure();
  ++begin;
  for (; begin != end; ++begin) {
    betweenFn();
    if (failed(eachFn(*begin)))
      return failure();
  }
  return success();
}

template <typename Container, typename UnaryFunctor, typename NullaryFunctor>
inline LogicalResult interleaveWithError(const Container &c,
                                         UnaryFunctor eachFn,
                                         NullaryFunctor betweenFn) {
  return interleaveWithError(c.begin(), c.end(), eachFn, betweenFn);
}

template <typename Container, typename UnaryFunctor>
inline LogicalResult interleaveCommaWithError(const Container &c,
                                              raw_ostream &os,
                                              UnaryFunctor eachFn) {
  return interleaveWithError(c.begin(), c.end(), eachFn, [&]() { os << ", "; });
}

namespace {
/// Emitter that uses dialect specific emitters to emit C++ code.
struct PythonEmitter {
  explicit PythonEmitter(raw_ostream &os, bool declareVariablesAtTop);

  /// Emits attribute or returns failure.
  LogicalResult emitAttribute(Location loc, Attribute attr);

  /// Emits operation 'op' with/without training semicolon or returns failure.
  LogicalResult emitOperation(Operation &op, bool trailingSemicolon);

  /// Emits type 'type' or returns failure.
  LogicalResult emitType(Location loc, Type type);

  /// Emits array of types as a std::tuple of the emitted types.
  /// - emits void for an empty array;
  /// - emits the type of the only element for arrays of size one;
  /// - emits a std::tuple otherwise;
  LogicalResult emitTypes(Location loc, ArrayRef<Type> types);

  /// Emits array of types as a std::tuple of the emitted types independently of
  /// the array size.
  LogicalResult emitTupleType(Location loc, ArrayRef<Type> types);

  /// Emits an assignment for a variable which has been declared previously.
  LogicalResult emitVariableAssignment(OpResult result);

  /// Emits a variable declaration for a result of an operation.
  LogicalResult emitVariableDeclaration(OpResult result,
                                        bool trailingSemicolon);

  /// Emits the variable declaration and assignment prefix for 'op'.
  /// - emits separate variable followed by std::tie for multi-valued operation;
  /// - emits single type followed by variable for single result;
  /// - emits nothing if no value produced by op;
  /// Emits final '=' operator where a type is produced. Returns failure if
  /// any result type could not be converted.
  LogicalResult emitAssignPrefix(Operation &op);

  /// Emits a label for the block.
  LogicalResult emitLabel(Block &block);

  /// Emits the operands and atttributes of the operation. All operands are
  /// emitted first and then all attributes in alphabetical order.
  LogicalResult emitOperandsAndAttributes(Operation &op,
                                          ArrayRef<StringRef> exclude = {});

  /// Emits the operands of the operation. All operands are emitted in order.
  LogicalResult emitOperands(Operation &op);

  /// Return the existing or a new name for a Value.
  StringRef getOrCreateName(Value val);

  /// Return the existing or a new label of a Block.
  StringRef getOrCreateName(Block &block);

  /// Whether to map an mlir integer to a unsigned integer in C++.
  bool shouldMapToUnsigned(IntegerType::SignednessSemantics val);

  /// RAII helper function to manage entering/exiting C++ scopes.
  struct Scope {
    Scope(PythonEmitter &emitter)
        : valueMapperScope(emitter.valueMapper),
          blockMapperScope(emitter.blockMapper), emitter(emitter) {
      emitter.valueInScopeCount.push(emitter.valueInScopeCount.top());
      emitter.labelInScopeCount.push(emitter.labelInScopeCount.top());
    }
    ~Scope() {
      emitter.valueInScopeCount.pop();
      emitter.labelInScopeCount.pop();
    }

  private:
    llvm::ScopedHashTableScope<Value, std::string> valueMapperScope;
    llvm::ScopedHashTableScope<Block *, std::string> blockMapperScope;
    PythonEmitter &emitter;
  };

  /// Returns wether the Value is assigned to a C++ variable in the scope.
  bool hasValueInScope(Value val);

  // Returns whether a label is assigned to the block.
  bool hasBlockLabel(Block &block);

  /// Returns the output stream.
  raw_indented_ostream &ostream() { return os; };

  /// Returns if all variables for op results and basic block arguments need to
  /// be declared at the beginning of a function.
  bool shouldDeclareVariablesAtTop() { return declareVariablesAtTop; };

private:
  using ValueMapper = llvm::ScopedHashTable<Value, std::string>;
  using BlockMapper = llvm::ScopedHashTable<Block *, std::string>;

  /// Output stream to emit to.
  raw_indented_ostream os;

  /// Boolean to enforce that all variables for op results and block
  /// arguments are declared at the beginning of the function. This also
  /// includes results from ops located in nested regions.
  bool declareVariablesAtTop;

  /// Map from value to name of C++ variable that contain the name.
  ValueMapper valueMapper;

  /// Map from block to name of C++ label.
  BlockMapper blockMapper;

  /// The number of values in the current scope. This is used to declare the
  /// names of values in a scope.
  std::stack<int64_t> valueInScopeCount;
  std::stack<int64_t> labelInScopeCount;
};
} // namespace

static LogicalResult printConstantOp(PythonEmitter &emitter,
                                     Operation *operation, Attribute value) {
  OpResult result = operation->getResult(0);

  // Only emit an assignment as the variable was already declared when printing
  // the FuncOp.
  if (emitter.shouldDeclareVariablesAtTop()) {
    if (failed(emitter.emitVariableAssignment(result)))
      return failure();
    return emitter.emitAttribute(operation->getLoc(), value);
  }

  // Emit a variable declaration.
  if (failed(emitter.emitAssignPrefix(*operation)))
    return failure();
  return emitter.emitAttribute(operation->getLoc(), value);
}

static LogicalResult printOperation(PythonEmitter &emitter,
                                    arith::ConstantOp constantOp) {
  Operation *operation = constantOp.getOperation();
  Attribute value = constantOp.getValue();

  return printConstantOp(emitter, operation, value);
}

static LogicalResult printOperation(PythonEmitter &emitter,
                                    func::ConstantOp constantOp) {
  Operation *operation = constantOp.getOperation();
  Attribute value = constantOp.getValueAttr();

  return printConstantOp(emitter, operation, value);
}

static LogicalResult printBinaryArithOperation(PythonEmitter &emitter,
                                               Operation *operation,
                                               StringRef binaryArithOperator) {
  raw_ostream &os = emitter.ostream();

  if (failed(emitter.emitAssignPrefix(*operation)))
    return failure();
  os << emitter.getOrCreateName(operation->getOperand(0));
  os << " " << binaryArithOperator;
  os << " " << emitter.getOrCreateName(operation->getOperand(1));

  return success();
}

static LogicalResult printOperation(PythonEmitter &emitter,
                                    arith::AddIOp addiOp) {
  Operation *operation = addiOp.getOperation();

  return printBinaryArithOperation(emitter, operation, "+");
}

static LogicalResult printOperation(PythonEmitter &emitter,
                                    arith::MulIOp muliOp) {
  Operation *operation = muliOp.getOperation();

  return printBinaryArithOperation(emitter, operation, "*");
}

static LogicalResult printOperation(PythonEmitter &emitter,
                                    arith::AddFOp addfOp) {
  Operation *operation = addfOp.getOperation();

  return printBinaryArithOperation(emitter, operation, "+");
}

static LogicalResult printOperation(PythonEmitter &emitter,
                                    arith::SubFOp subfOp) {
  Operation *operation = subfOp.getOperation();

  return printBinaryArithOperation(emitter, operation, "-");
}

static LogicalResult printOperation(PythonEmitter &emitter,
                                    arith::MulFOp mulfOp) {
  Operation *operation = mulfOp.getOperation();

  return printBinaryArithOperation(emitter, operation, "*");
}

static LogicalResult printOperation(PythonEmitter &emitter,
                                    arith::DivFOp divfOp) {
  Operation *operation = divfOp.getOperation();

  return printBinaryArithOperation(emitter, operation, "/");
}

static LogicalResult printOperation(PythonEmitter &emitter,
                                    arith::RemFOp remfOp) {
  Operation *operation = remfOp.getOperation();

  return printBinaryArithOperation(emitter, operation, "%");
}

static LogicalResult printOperation(PythonEmitter &emitter,
                                    arith::MaxFOp maxfOp) {
  Operation *operation = maxfOp.getOperation();

  raw_ostream &os = emitter.ostream();

  if (failed(emitter.emitAssignPrefix(*operation)))
    return failure();
  os << " ";

  os << "Max(";
  os << emitter.getOrCreateName(operation->getOperand(0)) << ", ";
  os << emitter.getOrCreateName(operation->getOperand(1)) << ")";

  return success();
}

static LogicalResult printCompareArithOperation(PythonEmitter &emitter,
                                                Operation *operation,
                                                StringRef comparator) {
  raw_ostream &os = emitter.ostream();

  if (failed(emitter.emitAssignPrefix(*operation)))
    return failure();

  os << "(";
  os << emitter.getOrCreateName(operation->getOperand(0));
  os << " " << comparator << " ";
  os << emitter.getOrCreateName(operation->getOperand(1));
  os << ")";

  return success();
}

static LogicalResult printOperation(PythonEmitter &emitter,
                                    arith::CmpIOp cmpiOp) {
  Operation *operation = cmpiOp.getOperation();
  arith::CmpIPredicate pred = cmpiOp.getPredicate();

  switch (pred) {
  case arith::CmpIPredicate::eq:
    return printCompareArithOperation(emitter, operation, "==");
  case arith::CmpIPredicate::sge:
    return printCompareArithOperation(emitter, operation, ">=");
  default:
    return operation->emitError("unsupported integer comparison predicate in "
                                "arith.cmpi operation: " +
                                std::to_string(static_cast<int>(pred)));
  }

  return success();
}

static LogicalResult printOperation(PythonEmitter &emitter,
                                    arith::CmpFOp cmpfOp) {
  Operation *operation = cmpfOp.getOperation();
  arith::CmpFPredicate pred = cmpfOp.getPredicate();

  switch (pred) {
  case arith::CmpFPredicate::OEQ:
  case arith::CmpFPredicate::UEQ: {
    if (failed(emitter.emitAssignPrefix(*cmpfOp.getOperation())))
      return failure();
    raw_ostream &os = emitter.ostream();
    os << "Eq(";
    if (failed(emitter.emitOperands(*cmpfOp.getOperation())))
      return failure();
    os << ")";
    return success();
  }

  case arith::CmpFPredicate::ONE:
  case arith::CmpFPredicate::UNE:
    return printCompareArithOperation(emitter, operation, "!=");
  case arith::CmpFPredicate::OGT:
    return printCompareArithOperation(emitter, operation, ">");
  case arith::CmpFPredicate::OGE:
    return printCompareArithOperation(emitter, operation, ">=");
  case arith::CmpFPredicate::OLT:
    return printCompareArithOperation(emitter, operation, "<");
  case arith::CmpFPredicate::OLE:
    return printCompareArithOperation(emitter, operation, "<=");
  default:
    return operation->emitError("unsupported integer comparison predicate in "
                                "arith.cmpf operation: " +
                                std::to_string(static_cast<int>(pred)));
  }

  return success();
}

static LogicalResult printOperation(PythonEmitter &emitter,
                                    arith::AndIOp andiOp) {
  Operation *operation = andiOp.getOperation();

  raw_ostream &os = emitter.ostream();

  if (failed(emitter.emitAssignPrefix(*operation)))
    return failure();

  os << "(";
  os << emitter.getOrCreateName(operation->getOperand(0));
  os << " & ";
  os << emitter.getOrCreateName(operation->getOperand(1));
  os << ")";

  return success();
}

static LogicalResult printOperation(PythonEmitter &emitter,
                                    arith::IndexCastOp idxCastOp) {
  Operation *operation = idxCastOp.getOperation();

  raw_ostream &os = emitter.ostream();

  for (auto result : operation->getResults()) {
    // os << "size_t ";
    os << emitter.getOrCreateName(result) << " = ";
    os << emitter.getOrCreateName(idxCastOp.getOperand());
  }

  return success();
}

static LogicalResult printOperation(PythonEmitter &emitter,
                                    arith::SelectOp selectOp) {
  Operation *operation = selectOp.getOperation();

  raw_ostream &os = emitter.ostream();

  for (auto result : operation->getResults()) {
    if (failed(emitter.emitVariableDeclaration(result,
                                               /*trailingSemicolon=*/false)))
      return failure();

    os << " = ";
    // os << emitter.getOrCreateName(selectOp.getTrueValue()) << " if ";
    // os << emitter.getOrCreateName(selectOp.getCondition()) << " else ";
    // os << emitter.getOrCreateName(selectOp.getFalseValue());

    os << "Piecewise(";
    os << "(" << emitter.getOrCreateName(selectOp.getTrueValue()) << ", "
       << emitter.getOrCreateName(selectOp.getCondition()) << "), ";
    os << "(" << emitter.getOrCreateName(selectOp.getFalseValue()) << ", "
       << "True))";
  }

  return success();
}

static LogicalResult printOperation(PythonEmitter &emitter,
                                    arith::FPToSIOp op) {
  Operation *operation = op.getOperation();

  raw_ostream &os = emitter.ostream();

  if (failed(emitter.emitAssignPrefix(*operation)))
    return failure();
  os << " ";
  os << emitter.getOrCreateName(operation->getOperand(0));

  return success();
}

static LogicalResult printOperation(PythonEmitter &emitter,
                                    arith::SIToFPOp op) {
  Operation *operation = op.getOperation();

  raw_ostream &os = emitter.ostream();

  if (failed(emitter.emitAssignPrefix(*operation)))
    return failure();
  os << " ";
  os << emitter.getOrCreateName(operation->getOperand(0));

  return success();
}

static LogicalResult printOperation(PythonEmitter &emitter,
                                    arith::UIToFPOp op) {
  Operation *operation = op.getOperation();
  raw_ostream &os = emitter.ostream();

  if (failed(emitter.emitAssignPrefix(*operation)))
    return failure();
  os << " ";

  // Get bitwith of the source type. If it is 1, we need to insert a Piecewise
  auto srcType = operation->getOperand(0).getType().cast<IntegerType>();
  auto srcWidth = srcType.getWidth();
  if (srcWidth == 1) {
    os << "Piecewise(";
    os << "(1.0, " << emitter.getOrCreateName(operation->getOperand(0))
       << "), ";
    os << "(0.0, True))";

    return success();
  }

  os << emitter.getOrCreateName(operation->getOperand(0));

  return success();
}

static LogicalResult printOperation(PythonEmitter &emitter,
                                    cf::BranchOp branchOp) {
  raw_ostream &os = emitter.ostream();
  Block &successor = *branchOp.getSuccessor();

  for (auto pair :
       llvm::zip(branchOp.getOperands(), successor.getArguments())) {
    Value &operand = std::get<0>(pair);
    BlockArgument &argument = std::get<1>(pair);
    os << emitter.getOrCreateName(argument) << " = "
       << emitter.getOrCreateName(operand) << ";\n";
  }

  os << "goto ";
  if (!(emitter.hasBlockLabel(successor)))
    return branchOp.emitOpError("unable to find label for successor block");
  os << emitter.getOrCreateName(successor);
  return success();
}

static LogicalResult printOperation(PythonEmitter &emitter,
                                    cf::CondBranchOp condBranchOp) {
  raw_indented_ostream &os = emitter.ostream();
  Block &trueSuccessor = *condBranchOp.getTrueDest();
  Block &falseSuccessor = *condBranchOp.getFalseDest();

  os << "if (" << emitter.getOrCreateName(condBranchOp.getCondition())
     << ") {\n";

  os.indent();

  // If condition is true.
  for (auto pair : llvm::zip(condBranchOp.getTrueOperands(),
                             trueSuccessor.getArguments())) {
    Value &operand = std::get<0>(pair);
    BlockArgument &argument = std::get<1>(pair);
    os << emitter.getOrCreateName(argument) << " = "
       << emitter.getOrCreateName(operand) << ";\n";
  }

  os << "goto ";
  if (!(emitter.hasBlockLabel(trueSuccessor))) {
    return condBranchOp.emitOpError("unable to find label for successor block");
  }
  os << emitter.getOrCreateName(trueSuccessor) << ";\n";
  os.unindent() << "} else {\n";
  os.indent();
  // If condition is false.
  for (auto pair : llvm::zip(condBranchOp.getFalseOperands(),
                             falseSuccessor.getArguments())) {
    Value &operand = std::get<0>(pair);
    BlockArgument &argument = std::get<1>(pair);
    os << emitter.getOrCreateName(argument) << " = "
       << emitter.getOrCreateName(operand) << ";\n";
  }

  os << "goto ";
  if (!(emitter.hasBlockLabel(falseSuccessor))) {
    return condBranchOp.emitOpError()
           << "unable to find label for successor block";
  }
  os << emitter.getOrCreateName(falseSuccessor) << ";\n";
  os.unindent() << "}";
  return success();
}

static LogicalResult printOperation(PythonEmitter &emitter,
                                    func::CallOp callOp) {
  if (failed(emitter.emitAssignPrefix(*callOp.getOperation())))
    return failure();

  raw_ostream &os = emitter.ostream();
  os << callOp.getCallee() << "(";
  if (failed(emitter.emitOperands(*callOp.getOperation())))
    return failure();
  os << ")";
  return success();
}

static LogicalResult printOperation(PythonEmitter &emitter, scf::ForOp forOp) {

  raw_indented_ostream &os = emitter.ostream();

  OperandRange operands = forOp.getIterOperands();
  Block::BlockArgListType iterArgs = forOp.getRegionIterArgs();
  Operation::result_range results = forOp.getResults();

  if (!emitter.shouldDeclareVariablesAtTop()) {
    for (OpResult result : results) {
      if (failed(emitter.emitVariableDeclaration(result,
                                                 /*trailingSemicolon=*/true)))
        return failure();
    }
  }

  for (auto pair : llvm::zip(iterArgs, operands)) {
    if (failed(emitter.emitType(forOp.getLoc(), std::get<0>(pair).getType())))
      return failure();
    os << " " << emitter.getOrCreateName(std::get<0>(pair)) << " = ";
    os << emitter.getOrCreateName(std::get<1>(pair)) << ";";
    os << "\n";
  }

  os << "for ";
  os << emitter.getOrCreateName(forOp.getInductionVar());
  os << " in range(";
  os << emitter.getOrCreateName(forOp.getLowerBound());
  os << ", ";
  os << emitter.getOrCreateName(forOp.getUpperBound());
  os << ", ";
  os << emitter.getOrCreateName(forOp.getStep());
  os << "):\n";
  os.indent();

  Region &forRegion = forOp.getRegion();
  auto regionOps = forRegion.getOps();

  // We skip the trailing yield op because this updates the result variables
  // of the for op in the generated code. Instead we update the iterArgs at
  // the end of a loop iteration and set the result variables after the for
  // loop.
  for (auto it = regionOps.begin(); std::next(it) != regionOps.end(); ++it) {
    if (failed(emitter.emitOperation(*it, /*trailingSemicolon=*/true)))
      return failure();
  }

  Operation *yieldOp = forRegion.getBlocks().front().getTerminator();
  // Copy yield operands into iterArgs at the end of a loop iteration.
  for (auto pair : llvm::zip(iterArgs, yieldOp->getOperands())) {
    BlockArgument iterArg = std::get<0>(pair);
    Value operand = std::get<1>(pair);
    os << emitter.getOrCreateName(iterArg) << " = "
       << emitter.getOrCreateName(operand) << ";\n";
  }

  os.unindent();

  // Copy iterArgs into results after the for loop.
  for (auto pair : llvm::zip(results, iterArgs)) {
    OpResult result = std::get<0>(pair);
    BlockArgument iterArg = std::get<1>(pair);
    os << "\n"
       << emitter.getOrCreateName(result) << " = "
       << emitter.getOrCreateName(iterArg) << ";";
  }

  return success();
}

static LogicalResult printOperation(PythonEmitter &emitter, scf::IfOp ifOp) {
  raw_indented_ostream &os = emitter.ostream();

  if (!emitter.shouldDeclareVariablesAtTop()) {
    for (OpResult result : ifOp.getResults()) {
      if (failed(emitter.emitVariableDeclaration(result,
                                                 /*trailingSemicolon=*/true)))
        return failure();
    }
  }

  os << "if (";
  if (failed(emitter.emitOperands(*ifOp.getOperation())))
    return failure();
  os << ") {\n";
  os.indent();

  Region &thenRegion = ifOp.getThenRegion();
  for (Operation &op : thenRegion.getOps()) {
    // Note: This prints a superfluous semicolon if the terminating yield op
    // has zero results.
    if (failed(emitter.emitOperation(op, /*trailingSemicolon=*/true)))
      return failure();
  }

  os.unindent() << "}";

  Region &elseRegion = ifOp.getElseRegion();
  if (!elseRegion.empty()) {
    os << " else {\n";
    os.indent();

    for (Operation &op : elseRegion.getOps()) {
      // Note: This prints a superfluous semicolon if the terminating yield op
      // has zero results.
      if (failed(emitter.emitOperation(op, /*trailingSemicolon=*/true)))
        return failure();
    }

    os.unindent() << "}";
  }

  return success();
}

static LogicalResult printOperation(PythonEmitter &emitter,
                                    scf::YieldOp yieldOp) {
  raw_ostream &os = emitter.ostream();
  Operation &parentOp = *yieldOp.getOperation()->getParentOp();

  if (yieldOp.getNumOperands() != parentOp.getNumResults()) {
    return yieldOp.emitError("number of operands does not to match the number "
                             "of the parent op's results");
  }

  if (failed(interleaveWithError(
          llvm::zip(parentOp.getResults(), yieldOp.getOperands()),
          [&](auto pair) -> LogicalResult {
            auto result = std::get<0>(pair);
            auto operand = std::get<1>(pair);
            os << emitter.getOrCreateName(result) << " = ";

            if (!emitter.hasValueInScope(operand))
              return yieldOp.emitError("operand value not in scope");
            os << emitter.getOrCreateName(operand);
            return success();
          },
          [&]() { os << ";\n"; })))
    return failure();

  return success();
}

static LogicalResult printOperation(PythonEmitter &emitter,
                                    math::SqrtOp sqrtOp) {
  // Emit a variable declaration.
  if (failed(emitter.emitAssignPrefix(*sqrtOp)))
    return failure();

  raw_ostream &os = emitter.ostream();
  os << "sqrt(";

  if (failed(emitter.emitOperands(*sqrtOp.getOperation())))
    return failure();

  os << ")";

  return success();
}

static LogicalResult printOperation(PythonEmitter &emitter,
                                    math::PowFOp powfOp) {
  Operation *operation = powfOp.getOperation();
  return printBinaryArithOperation(emitter, operation, "**");
}

static LogicalResult printOperation(PythonEmitter &emitter, math::ExpOp expOp) {
  // Emit a variable declaration.
  if (failed(emitter.emitAssignPrefix(*expOp)))
    return failure();

  raw_ostream &os = emitter.ostream();
  os << "exp(";

  if (failed(emitter.emitOperands(*expOp.getOperation())))
    return failure();

  os << ")";

  return success();
}

void emitMemrefAllocation(PythonEmitter &emitter, OpResult res) {
    if (auto mType = dyn_cast<MemRefType>(res.getType())) {
      std::string curStr = "None";

      auto shape = mType.getShape();
      for (auto dimSize : llvm::reverse(shape)) {
        curStr =
            "[" + curStr + " for _ in range(" + std::to_string(dimSize) + ")]";
      }

      emitter.ostream() << " = " << curStr;
    }
}

static LogicalResult printOperation(PythonEmitter &emitter,
                                    memref::AllocOp allocOp) {
  for (auto res : allocOp->getResults()) {
    if (failed(emitter.emitVariableDeclaration(res,
                                               /*trailingSemicolon=*/false)))
      return failure();
    emitMemrefAllocation(emitter, res);
  }
  return success();
}

static LogicalResult printOperation(PythonEmitter &emitter,
                                    memref::AllocaOp allocOp) {
  for (auto res : allocOp->getResults()) {
    if (failed(emitter.emitVariableDeclaration(res,
                                               /*trailingSemicolon=*/false)))
      return failure();
    emitMemrefAllocation(emitter, res);
  }

  return success();
}

static LogicalResult printOperation(PythonEmitter &emitter,
                                    memref::DeallocOp deallocOp) {
  return success();
}

static LogicalResult printOperation(PythonEmitter &emitter,
                                    memref::StoreOp storeOp) {
  raw_ostream &os = emitter.ostream();

  os << emitter.getOrCreateName(storeOp.getMemRef());
  for (auto indice : storeOp.getIndices()) {
    os << "[" << emitter.getOrCreateName(indice) << "]";
  }
  os << " = " << emitter.getOrCreateName(storeOp.getValue());

  return success();
}

static LogicalResult printOperation(PythonEmitter &emitter,
                                    memref::LoadOp loadOp) {
  raw_ostream &os = emitter.ostream();

  if (failed(emitter.emitAssignPrefix(*loadOp)))
    return failure();

  os << emitter.getOrCreateName(loadOp.getMemRef());
  for (auto indice : loadOp.getIndices()) {
    os << "[" << emitter.getOrCreateName(indice) << "]";
  }

  return success();
}

static LogicalResult printOperation(PythonEmitter &emitter,
                                    memref::GlobalOp globalOp) {
  raw_ostream &os = emitter.ostream();

  os << globalOp.getSymName() << " = ";

  auto getFloat = [&](const APFloat &val) -> std::string {
    if (val.isFinite()) {
      SmallString<128> strValue;
      // Use default values of toString except don't truncate zeros.
      val.toString(strValue, 0, 0, false);
      return std::string(strValue);
    } else if (val.isNaN()) {
      return "float('NaN')";
    } else if (val.isInfinity()) {
      if (val.isNegative())
        return "float('-inf')";
      return "float('inf')";
    }
  };

  mlir::SmallVector<std::string> elements;
  if (auto dType =
          dyn_cast<DenseElementsAttr>(globalOp.getInitialValueAttr())) {

    // Check if values are FloatAttr.
    if (isa<FloatType>(dType.getType().getElementType())) {
      for (auto element : dType.getValues<FloatAttr>()) {
        elements.push_back(getFloat(element.getValue()));
      }
    } else if (isa<IntegerType>(dType.getType().getElementType())) {
      // Check if values are IntegerAttr.
      for (auto element : dType.getValues<IntegerAttr>()) {
        // Check if values are a i1.
        if (element.getType().isInteger(1)) {
          elements.push_back(
              std::to_string(element.getValue().getSExtValue() != 0));
        } else {
          elements.push_back(std::to_string(element.getValue().getSExtValue()));
        }
      }
    } else if (succeeded(dType.tryGetValues<BoolAttr>())) {
      for (auto element : dType.getValues<BoolAttr>()) {
        elements.push_back(std::to_string(element.getValue()));
      }
    } else {
      return globalOp.emitOpError("unsupported global type");
    }
  }

  if (auto mType = dyn_cast<MemRefType>(globalOp.getType())) {
    auto shape = mType.getShape();

    if (shape.empty()) {
      os << elements[0];
    } else if (shape.size() == 1) {
      os << "[" << llvm::join(elements, ", ") << "]";
    } else if (shape.size() == 2) {
      os << "[";
      for (int r = 0; r < shape[0]; r++) {
        os << "[";
        for (int c = 0; c < shape[1]; c++) {
          os << elements[r * shape[1] + c];
          if (c < shape[1] - 1)
            os << ", ";
        }
        os << "]";
        if (r < shape[0] - 1)
          os << ", ";
      }
      os << "]";
    } else if (shape.size() == 3) {
      os << "[";
      for (int r = 0; r < shape[0]; r++) {
        os << "[";
        for (int c = 0; c < shape[1]; c++) {
          os << "[";
          for (int d = 0; d < shape[2]; d++) {
            os << elements[r * shape[1] * shape[2] + c * shape[2] + d];
            if (d < shape[2] - 1)
              os << ", ";
          }
          os << "]";
          if (c < shape[1] - 1)
            os << ", ";
        }
        os << "]";
        if (r < shape[0] - 1)
          os << ", ";
      }
      os << "]";
    } else {
      return globalOp.emitOpError("unsupported global shape");
    }
  }

  return success();
}

static LogicalResult printOperation(PythonEmitter &emitter,
                                    memref::GetGlobalOp globalOp) {
  raw_ostream &os = emitter.ostream();

  os << emitter.getOrCreateName(globalOp.getResult());
  os << " = ";
  os << globalOp.getName();

  return success();
}

static LogicalResult printOperation(PythonEmitter &emitter,
                                    func::ReturnOp returnOp) {
  raw_ostream &os = emitter.ostream();
  os << "return";
  switch (returnOp.getNumOperands()) {
  case 0:
    os << "\n";
    return success();
  case 1:
    os << " " << emitter.getOrCreateName(returnOp.getOperand(0));
    os << "\n";
    return success(emitter.hasValueInScope(returnOp.getOperand(0)));
  default:
    os << " std::make_tuple(";
    if (failed(emitter.emitOperandsAndAttributes(*returnOp.getOperation())))
      return failure();
    os << ")";
    os << "\n";
    return success();
  }
}

static LogicalResult printOperation(PythonEmitter &emitter, ModuleOp moduleOp) {
  PythonEmitter::Scope scope(emitter);

  for (Operation &op : moduleOp) {
    if (failed(emitter.emitOperation(op, /*trailingSemicolon=*/false)))
      return failure();
  }
  return success();
}

static LogicalResult printOperation(PythonEmitter &emitter,
                                    func::FuncOp functionOp) {
  // We need to declare variables at top if the function has multiple blocks.
  if (!emitter.shouldDeclareVariablesAtTop() &&
      functionOp.getBlocks().size() > 1) {
    return functionOp.emitOpError(
        "with multiple blocks needs variables declared at top");
  }

  PythonEmitter::Scope scope(emitter);
  raw_indented_ostream &os = emitter.ostream();

  //// Function types.
  // os << "# FnArgTypes: [";
  // if (failed(interleaveCommaWithError(
  //         functionOp.getArguments(), os,
  //         [&](BlockArgument arg) -> LogicalResult {
  //           os << "\"";
  //           if (failed(emitter.emitType(functionOp.getLoc(), arg.getType())))
  //             return failure();
  //           os << "\"";
  //           return success();
  //         })))
  //   return failure();
  // os << "]\n";

  // os << "# FnArgIdxs: [";
  // for (unsigned argIdx = 0; argIdx < functionOp.getNumArguments(); argIdx++)
  // {
  //   if (auto fnArgIdxAttr = functionOp.getArgAttr(argIdx,
  //   "irsynth.fnArgIdx")) {
  //     auto fnArgIdx = fnArgIdxAttr.dyn_cast<IntegerAttr>();
  //     os << fnArgIdx.getInt();
  //     if (argIdx < functionOp.getNumArguments() - 1)
  //       os << ", ";
  //   }
  // }
  // os << "]\n";
  // os << "\n";

  // Function.
  // os << "def " << functionOp.getName();
  os << "def foo";

  os << "(";
  if (failed(interleaveCommaWithError(functionOp.getArguments(), os,
                                      [&](BlockArgument arg) -> LogicalResult {
                                        os << emitter.getOrCreateName(arg);
                                        return success();
                                      })))
    return failure();
  os << "):";

  os << "\n";
  os.indent();
  if (emitter.shouldDeclareVariablesAtTop()) {
    // Declare all variables that hold op results including those from nested
    // regions.
    WalkResult result =
        functionOp.walk<WalkOrder::PreOrder>([&](Operation *op) -> WalkResult {
          for (OpResult result : op->getResults()) {
            if (failed(emitter.emitVariableDeclaration(
                    result, /*trailingSemicolon=*/true))) {
              return WalkResult(
                  op->emitError("unable to declare result variable for op"));
            }
          }
          return WalkResult::advance();
        });
    if (result.wasInterrupted())
      return failure();
  }

  Region::BlockListType &blocks = functionOp.getBlocks();
  // Create label names for basic blocks.
  for (Block &block : blocks) {
    emitter.getOrCreateName(block);
  }

  // Declare variables for basic block arguments.
  for (Block &block : llvm::drop_begin(blocks)) {
    for (BlockArgument &arg : block.getArguments()) {
      if (emitter.hasValueInScope(arg))
        return functionOp.emitOpError(" block argument #")
               << arg.getArgNumber() << " is out of scope";
      if (failed(
              emitter.emitType(block.getParentOp()->getLoc(), arg.getType()))) {
        return failure();
      }
      os << " " << emitter.getOrCreateName(arg) << ";\n";
    }
  }

  for (Block &block : blocks) {
    // Only print a label if the block has predecessors.
    if (!block.hasNoPredecessors()) {
      if (failed(emitter.emitLabel(block)))
        return failure();
    }
    for (Operation &op : block.getOperations()) {
      // When generating code for an scf.if or cf.cond_br op no semicolon
      // needs to be printed after the closing brace. When generating code for
      // an scf.for op, printing a trailing semicolon is handled within the
      // printOperation function.
      bool trailingSemicolon =
          !isa<scf::IfOp, scf::ForOp, cf::CondBranchOp>(op);

      if (failed(emitter.emitOperation(
              op, /*trailingSemicolon=*/trailingSemicolon)))
        return failure();
    }
  }
  os.unindent();
  return success();
}

PythonEmitter::PythonEmitter(raw_ostream &os, bool declareVariablesAtTop)
    : os(os), declareVariablesAtTop(declareVariablesAtTop) {
  valueInScopeCount.push(0);
  labelInScopeCount.push(0);
}

/// Return the existing or a new name for a Value.
StringRef PythonEmitter::getOrCreateName(Value val) {
  if (!valueMapper.count(val))
    valueMapper.insert(val, "v" + std::to_string(++valueInScopeCount.top()));
  return *valueMapper.begin(val);
}

/// Return the existing or a new label for a Block.
StringRef PythonEmitter::getOrCreateName(Block &block) {
  if (!blockMapper.count(&block))
    blockMapper.insert(&block, "v" + std::to_string(++labelInScopeCount.top()));
  return *blockMapper.begin(&block);
}

bool PythonEmitter::shouldMapToUnsigned(IntegerType::SignednessSemantics val) {
  switch (val) {
  case IntegerType::Signless:
    return false;
  case IntegerType::Signed:
    return false;
  case IntegerType::Unsigned:
    return true;
  }
  llvm_unreachable("Unexpected IntegerType::SignednessSemantics");
}

bool PythonEmitter::hasValueInScope(Value val) {
  return valueMapper.count(val);
}

bool PythonEmitter::hasBlockLabel(Block &block) {
  return blockMapper.count(&block);
}

LogicalResult PythonEmitter::emitAttribute(Location loc, Attribute attr) {
  auto printInt = [&](const APInt &val, bool isUnsigned) {
    if (val.getBitWidth() == 1) {
      if (val.getBoolValue())
        os << "true";
      else
        os << "false";
    } else {
      SmallString<128> strValue;
      val.toString(strValue, 10, !isUnsigned, false);
      os << strValue;
    }
  };

  auto printFloat = [&](const APFloat &val) {
    if (val.isFinite()) {
      SmallString<128> strValue;
      // Use default values of toString except don't truncate zeros.
      val.toString(strValue, 0, 0, false);
      os << strValue;
    } else if (val.isNaN()) {
      os << "float('NaN')";
    } else if (val.isInfinity()) {
      if (val.isNegative())
        os << "float('-inf')";
      else
        os << "float('inf')";
    }
  };

  // Print floating point attributes.
  if (auto fAttr = dyn_cast<FloatAttr>(attr)) {
    printFloat(fAttr.getValue());
    return success();
  }
  if (auto dense = dyn_cast<DenseFPElementsAttr>(attr)) {
    os << '{';
    interleaveComma(dense, os, [&](const APFloat &val) { printFloat(val); });
    os << '}';
    return success();
  }

  // Print integer attributes.
  if (auto iAttr = dyn_cast<IntegerAttr>(attr)) {
    if (auto iType = dyn_cast<IntegerType>(iAttr.getType())) {
      printInt(iAttr.getValue(), shouldMapToUnsigned(iType.getSignedness()));
      return success();
    }
    if (auto iType = dyn_cast<IndexType>(iAttr.getType())) {
      printInt(iAttr.getValue(), false);
      return success();
    }
  }
  if (auto dense = dyn_cast<DenseIntElementsAttr>(attr)) {
    if (auto iType = dyn_cast<IntegerType>(
            cast<TensorType>(dense.getType()).getElementType())) {
      os << '{';
      interleaveComma(dense, os, [&](const APInt &val) {
        printInt(val, shouldMapToUnsigned(iType.getSignedness()));
      });
      os << '}';
      return success();
    }
    if (auto iType = dyn_cast<IndexType>(
            cast<TensorType>(dense.getType()).getElementType())) {
      os << '{';
      interleaveComma(dense, os,
                      [&](const APInt &val) { printInt(val, false); });
      os << '}';
      return success();
    }
  }

  // Print symbolic reference attributes.
  if (auto sAttr = dyn_cast<SymbolRefAttr>(attr)) {
    if (sAttr.getNestedReferences().size() > 1)
      return emitError(loc, "attribute has more than 1 nested reference");
    os << sAttr.getRootReference().getValue();
    return success();
  }

  // Print type attributes.
  if (auto type = dyn_cast<TypeAttr>(attr))
    return emitType(loc, type.getValue());

  return emitError(loc, "cannot emit attribute: ") << attr;
}

LogicalResult PythonEmitter::emitOperands(Operation &op) {
  auto emitOperandName = [&](Value result) -> LogicalResult {
    if (!hasValueInScope(result))
      return op.emitOpError() << "operand value not in scope";
    os << getOrCreateName(result);
    return success();
  };
  return interleaveCommaWithError(op.getOperands(), os, emitOperandName);
}

LogicalResult
PythonEmitter::emitOperandsAndAttributes(Operation &op,
                                         ArrayRef<StringRef> exclude) {
  if (failed(emitOperands(op)))
    return failure();
  // Insert comma in between operands and non-filtered attributes if needed.
  if (op.getNumOperands() > 0) {
    for (NamedAttribute attr : op.getAttrs()) {
      if (!llvm::is_contained(exclude, attr.getName().strref())) {
        os << ", ";
        break;
      }
    }
  }
  // Emit attributes.
  auto emitNamedAttribute = [&](NamedAttribute attr) -> LogicalResult {
    if (llvm::is_contained(exclude, attr.getName().strref()))
      return success();
    os << "/* " << attr.getName().getValue() << " */";
    if (failed(emitAttribute(op.getLoc(), attr.getValue())))
      return failure();
    return success();
  };
  return interleaveCommaWithError(op.getAttrs(), os, emitNamedAttribute);
}

LogicalResult PythonEmitter::emitVariableAssignment(OpResult result) {
  if (!hasValueInScope(result)) {
    return result.getDefiningOp()->emitOpError(
        "result variable for the operation has not been declared");
  }
  os << getOrCreateName(result) << " = ";
  return success();
}

LogicalResult PythonEmitter::emitVariableDeclaration(OpResult result,
                                                     bool trailingSemicolon) {
  if (hasValueInScope(result)) {
    return result.getDefiningOp()->emitError(
        "result variable for the operation already declared");
  }
  os << getOrCreateName(result);

  return success();
}

LogicalResult PythonEmitter::emitAssignPrefix(Operation &op) {
  switch (op.getNumResults()) {
  case 0:
    break;
  case 1: {
    OpResult result = op.getResult(0);
    if (shouldDeclareVariablesAtTop()) {
      if (failed(emitVariableAssignment(result)))
        return failure();
    } else {
      if (failed(emitVariableDeclaration(result, /*trailingSemicolon=*/false)))
        return failure();
      os << " = ";
    }
    break;
  }
  default:
    if (!shouldDeclareVariablesAtTop()) {
      for (OpResult result : op.getResults()) {
        if (failed(emitVariableDeclaration(result, /*trailingSemicolon=*/true)))
          return failure();
      }
    }
    os << "std::tie(";
    interleaveComma(op.getResults(), os,
                    [&](Value result) { os << getOrCreateName(result); });
    os << ") = ";
  }
  return success();
}

LogicalResult PythonEmitter::emitLabel(Block &block) {
  if (!hasBlockLabel(block))
    return block.getParentOp()->emitError("label for block not found");
  // FIXME: Add feature in `raw_indented_ostream` to ignore indent for block
  // label instead of using `getOStream`.
  os.getOStream() << getOrCreateName(block) << ":\n";
  return success();
}

LogicalResult PythonEmitter::emitOperation(Operation &op,
                                           bool trailingSemicolon) {
  LogicalResult status =
      llvm::TypeSwitch<Operation *, LogicalResult>(&op)
          // Builtin ops.
          .Case<ModuleOp>([&](auto op) { return printOperation(*this, op); })
          // CF ops.
          .Case<cf::BranchOp, cf::CondBranchOp>(
              [&](auto op) { return printOperation(*this, op); })
          // Func ops.
          .Case<func::CallOp, func::ConstantOp, func::FuncOp, func::ReturnOp>(
              [&](auto op) { return printOperation(*this, op); })
          // SCF ops.
          .Case<scf::ForOp, scf::IfOp, scf::YieldOp>(
              [&](auto op) { return printOperation(*this, op); })

          // Arithmetic ops.
          .Case<arith::ConstantOp>(
              [&](auto op) { return printOperation(*this, op); })
          .Case<arith::AddIOp, arith::MulIOp>(
              [&](auto op) { return printOperation(*this, op); })
          .Case<arith::AddFOp, arith::SubFOp, arith::MulFOp, arith::DivFOp>(
              [&](auto op) { return printOperation(*this, op); })
          .Case<arith::RemFOp>(
              [&](auto op) { return printOperation(*this, op); })
          .Case<arith::MaxFOp>(
              [&](auto op) { return printOperation(*this, op); })
          .Case<arith::CmpIOp, arith::CmpFOp>(
              [&](auto op) { return printOperation(*this, op); })
          .Case<arith::AndIOp>(
              [&](auto op) { return printOperation(*this, op); })
          .Case<arith::IndexCastOp>(
              [&](auto op) { return printOperation(*this, op); })
          .Case<arith::SelectOp>(
              [&](auto op) { return printOperation(*this, op); })
          .Case<arith::FPToSIOp, arith::SIToFPOp, arith::UIToFPOp>(
              [&](auto op) { return printOperation(*this, op); })

          // Math ops.
          .Case<math::SqrtOp>(
              [&](auto op) { return printOperation(*this, op); })
          .Case<math::PowFOp>(
              [&](auto op) { return printOperation(*this, op); })
          .Case<math::ExpOp>([&](auto op) { return printOperation(*this, op); })

          // Memref ops.
          .Case<memref::AllocOp, memref::DeallocOp, memref::AllocaOp>(
              [&](auto op) { return printOperation(*this, op); })
          .Case<memref::LoadOp, memref::StoreOp>(
              [&](auto op) { return printOperation(*this, op); })
          .Case<memref::GlobalOp, memref::GetGlobalOp>(
              [&](auto op) { return printOperation(*this, op); })

          .Default([&](Operation *) {
            return op.emitOpError("unable to find printer for op");
          });

  if (failed(status))
    return failure();
  if (!llvm::isa<func::FuncOp>(op) && !llvm::isa<func::ReturnOp>(op) &&
      !llvm::isa<scf::ForOp>(op))
    os << "\n";
  return success();
}

LogicalResult PythonEmitter::emitType(Location loc, Type type) {
  if (auto iType = dyn_cast<IntegerType>(type)) {
    switch (iType.getWidth()) {
    case 1:
      return (os << "bool"), success();
    case 8:
    case 16:
    case 32:
    case 64:
      if (shouldMapToUnsigned(iType.getSignedness()))
        return (os << "uint" << iType.getWidth() << "_t"), success();
      else
        return (os << "int" << iType.getWidth() << "_t"), success();
    default:
      return emitError(loc, "cannot emit integer type ") << type;
    }
  }
  if (auto fType = dyn_cast<FloatType>(type)) {
    switch (fType.getWidth()) {
    case 32:
      return (os << "float"), success();
    case 64:
      return (os << "double"), success();
    default:
      return emitError(loc, "cannot emit float type ") << type;
    }
  }
  if (auto iType = dyn_cast<IndexType>(type))
    return (os << "size_t"), success();
  if (auto tType = dyn_cast<TensorType>(type)) {
    if (!tType.hasRank())
      return emitError(loc, "cannot emit unranked tensor type");
    if (!tType.hasStaticShape())
      return emitError(loc, "cannot emit tensor type with non static shape");
    os << "Tensor<";
    if (failed(emitType(loc, tType.getElementType())))
      return failure();
    auto shape = tType.getShape();
    for (auto dimSize : shape) {
      os << ", ";
      os << dimSize;
    }
    os << ">";
    return success();
  }
  if (auto mType = dyn_cast<MemRefType>(type)) {
    if (failed(emitType(loc, mType.getElementType())))
      return failure();

    os << "[";
    for (auto [dimIdx, dimSize] : llvm::enumerate(mType.getShape())) {
      os << dimSize;
      if ((int64_t)dimIdx != mType.getRank() - 1)
        os << ",";
    }
    os << "]";

    return success();
  }
  if (auto tType = dyn_cast<TupleType>(type))
    return emitTupleType(loc, tType.getTypes());
  return emitError(loc, "cannot emit type ") << type;
}

LogicalResult PythonEmitter::emitTypes(Location loc, ArrayRef<Type> types) {
  switch (types.size()) {
  case 0:
    os << "void";
    return success();
  case 1:
    return emitType(loc, types.front());
  default:
    return emitTupleType(loc, types);
  }
}

LogicalResult PythonEmitter::emitTupleType(Location loc, ArrayRef<Type> types) {
  os << "std::tuple<";
  if (failed(interleaveCommaWithError(
          types, os, [&](Type type) { return emitType(loc, type); })))
    return failure();
  os << ">";
  return success();
}

LogicalResult translateToPython(Operation *op, raw_ostream &os,
                                bool declareVariablesAtTop) {
  PythonEmitter emitter(os, declareVariablesAtTop);
  return emitter.emitOperation(*op, /*trailingSemicolon=*/false);
}
