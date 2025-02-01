#ifndef IRSYNTH_SCOP_H
#define IRSYNTH_SCOP_H

#include "isl/isl_helper.h"

#include "mlir/Dialect/Affine/Analysis/AffineStructures.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <boost/graph/adjacency_list.hpp>
#include <unordered_map>

class ScopStmt {
public:
  ScopStmt(llvm::SmallVector<mlir::Operation *> allOps,
           llvm::SmallVector<mlir::Operation *> accessOps, std::string name)
      : allOps(allOps), accessOps(accessOps), name(name) {}

  isl::set domain() { return accessRelations[0].domain(); }

  void dump(llvm::raw_ostream &os, bool withLabels = true, bool withName = true,
            bool withDomain = true, bool withAccessOps = false);

public:
  llvm::SmallVector<mlir::Operation *> allOps;
  llvm::SmallVector<mlir::Operation *> accessOps;
  llvm::SmallVector<isl::map> accessRelations;
  std::string name;
};

class DependenceGraph {
public:
  class Node;
  using NodePtr = std::shared_ptr<Node>;
  class Node {
  public:
    Node(ScopStmt *stmt) : stmt(stmt) {}

    ScopStmt *stmt;
    std::vector<Node*> dependencies;
    std::vector<Node*> dependents;
  };

  void computeDependencies();
  void dump(llvm::raw_ostream &os);

  llvm::SmallVector<NodePtr> nodes;

private:
  int getNumDependencies();
};
using DependenceGraphPtr = std::shared_ptr<DependenceGraph>;

class Scop {
public:
  Scop(mlir::Operation *op);
  ~Scop();

  DependenceGraphPtr getDependenceGraph();

  ScopStmt *lookupStmtByName(std::string name);
  ScopStmt *lookupStmtByOp(mlir::Operation *op);
  llvm::SmallVector<ScopStmt> lookupStmtsByBlock(mlir::Block &block);

  void toDot(llvm::raw_ostream &os);
  void toDotStmts(llvm::raw_ostream &os);

  void dump(llvm::raw_ostream &os);

private:
  void buildScopStmts();
  void buildAccessRelationIslMaps();
  void computeFlowDependencies();

  isl::map getAccessRelationForOp(mlir::Operation *op, std::string &opName);

private:
  mlir::Operation *op;
  std::unordered_map<std::string, ScopStmt> namesToStmts;
  mlir::AsmState *asmState;
  isl::schedule schedule;
  isl::union_map flowDependencies;
  isl_ctx *ctx;
};
using ScopPtr = std::shared_ptr<Scop>;

void dumpRelDetails(mlir::FlatAffineRelation rel);

namespace {
struct PolyhedralAnalysisPass
    : public mlir::PassWrapper<PolyhedralAnalysisPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PolyhedralAnalysisPass)

  PolyhedralAnalysisPass() = default;
  PolyhedralAnalysisPass(const PolyhedralAnalysisPass &) {}

  Option<bool> dump{*this, "dump", llvm::cl::desc("Dump all info"),
                    llvm::cl::init(false)};

  Option<bool> dot{*this, "dot", llvm::cl::desc("Dot graph"),
                   llvm::cl::init(false)};

  Option<bool> dotStmts{*this, "dot-stmts",
                        llvm::cl::desc("Dot graph of statement contents"),
                        llvm::cl::init(false)};

  Option<bool> dumpDependenceGraph{*this, "dump-dependence-graph",
                                   llvm::cl::desc("Dump dependence graph"),
                                   llvm::cl::init(false)};

  llvm::StringRef getArgument() const override { return "polyhedral-analysis"; }
  llvm::StringRef getDescription() const override {
    return "Polyhedral dependence analysis using ISL.";
  }
  void runOnOperation() override {
    mlir::Operation *op = getOperation();
    scop = std::make_shared<Scop>(op);

    if (dump) {
      scop->dump(llvm::outs());
      llvm::outs() << "\n";
    }

    if (dot) {
      scop->toDot(llvm::outs());
      llvm::outs() << "\n";
    }

    if (dotStmts) {
      scop->toDotStmts(llvm::outs());
      llvm::outs() << "\n";
    }

    if (dumpDependenceGraph) {
      auto dg = scop->getDependenceGraph();
      dg->dump(llvm::outs());
      llvm::outs() << "\n";
    }
  }

  ScopPtr scop;
};
} // namespace

namespace mlir {
inline void registerPolyhedralAnalysisPass() { PassRegistration<PolyhedralAnalysisPass>(); }
} // namespace mlir

struct VertexInfoT {
  ScopStmt *stmt;
};
using BoostGraph = boost::adjacency_list<boost::vecS, boost::vecS,
                                         boost::directedS, VertexInfoT>;

namespace boost {
inline void throw_exception(std::exception const &e) {
  llvm::errs() << "Boost exception: " << e.what() << "\n";
  abort();
}
inline void throw_exception(std::exception const &e, boost::source_location const & /*loc*/) {
  llvm::errs() << "Boost exception: " << e.what() << "\n";
  abort();
}
} // namespace boost

BoostGraph constructBoostGraph(DependenceGraphPtr &dg);
void printBoostGraph(BoostGraph &g);
int computeNumCycles(BoostGraph const &g);

#endif // IRSYNTH_SCOP_H
