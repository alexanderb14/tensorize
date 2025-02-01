#include "PolyhedralAnalysis.h"

#include "isl/isl_tools.h"

#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/AffineStructures.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/AsmState.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"

#include <boost/graph/graph_traits.hpp>
#include <boost/graph/connected_components.hpp>
#include <boost/graph/dijkstra_shortest_paths.hpp>
#include <boost/graph/tiernan_all_cycles.hpp>

#include <regex>
#include <string>

using namespace llvm;
using namespace mlir;

// Create an isl_multi_union_aff that defines an identity mapping from the
// elements of USet to their N-th dimension.
//
// # Example: Domain: { A[i,j]; B[i,j,k] }, N: 1
// Resulting Mapping: { {A[i,j] -> [(j)]; B[i,j,k] -> [(j)] }
isl::multi_union_pw_aff mapToDimension(const isl::union_set &uSet, unsigned n) {
  // llvm::outs() << "mapToDimension " << stringFromIslObj(USet) << " " << N <<
  // "\n";
  assert(!uSet.is_null());
  assert(!uSet.is_empty());

  auto result = isl::union_pw_multi_aff::empty(uSet.get_space());

  for (isl::set s : uSet.get_set_list()) {
    unsigned dim = unsignedFromIslSize(s.tuple_dim());
    assert(dim >= n);
    auto pma = isl::pw_multi_aff::project_out_map(s.get_space(), isl::dim::set,
                                                  n, dim - n);
    if (n > 1)
      pma = pma.drop_dims(isl::dim::out, 0, n - 1);

    result = result.add_pw_multi_aff(pma);
  }

  // llvm::outs() << "MUPA: " <<
  // stringFromIslObj(isl::multi_union_pw_aff(isl::union_pw_multi_aff(Result)))
  // << "\n";
  return isl::multi_union_pw_aff(isl::union_pw_multi_aff(result));
}

isl::schedule combineInSequence(const isl::schedule &prev,
                                const isl::schedule &succ) {
  if (prev.is_null())
    return succ;
  if (succ.is_null())
    return prev;

  return prev.sequence(succ);
}

using ScopStmtMap = mlir::DenseMap<ScopStmt *, bool>;

isl::schedule scheduleBlock(ScopStmtMap *seenStmts, Block &block, Scop &scop, unsigned depth);
isl::schedule scheduleRegion(ScopStmtMap *seenStmts, Region &region, Scop &scop, unsigned depth);
isl::schedule scheduleOperation(ScopStmtMap *seenStmts, Operation *op,
                                Scop &scop, unsigned depth);

isl::schedule scheduleBlock(ScopStmtMap *seenStmts, Block &block, Scop &scop, unsigned depth) {
  isl::schedule sched;
  for (Operation &op : block.getOperations()) {
    unsigned depthNew = depth;
    if (isa<AffineForOp>(op))
      depthNew++;

    sched = combineInSequence(sched, scheduleOperation(seenStmts, &op, scop, depthNew));
  }

  return sched;
}

isl::schedule scheduleRegion(ScopStmtMap *seenStmts, Region &region, Scop &scop, unsigned depth) {
  isl::schedule sched;
  for (Block &block : region.getBlocks())
    sched = combineInSequence( sched, scheduleBlock(seenStmts, block, scop, depth));
  return sched;
}

isl::schedule scheduleOperation(ScopStmtMap *seenStmts, Operation *op,
                                Scop &scop, unsigned depth) {
  isl::schedule sched;

  for (Region &region : op->getRegions()) {
    sched = combineInSequence(sched, scheduleRegion(seenStmts, region, scop, depth));
  }

  ScopStmt *stmt = scop.lookupStmtByOp(op);
  if (stmt) {
    if (seenStmts->find(stmt) == seenStmts->end()) {
      (*seenStmts)[stmt] = true;

      isl::union_set domain;
      if (sched.is_null()) {
        domain = stmt->domain();
        sched = isl::schedule::from_domain(domain);
      } else {
        domain = sched.get_domain();
      }

      // isl::multi_union_pw_aff mupa = mapToDimension(domain, depth);
      // sched = sched.insert_partial_schedule(mupa);
    }

  } else if (isa<AffineForOp>(op)) {
    isl::multi_union_pw_aff mupa = mapToDimension(sched.get_domain(), depth);
    sched = sched.insert_partial_schedule(mupa);
  }

  return sched;
}

// ScopStmt
// ----------------------------------------
void ScopStmt::dump(raw_ostream &os, bool withLabels, bool withName,
                    bool withDomain, bool withAccessOps) {
  if (withName)
    os << name << "\n";

  if (withDomain) {
    if (withLabels)
      os << "- Domain\n";
    os << "    " << stringFromIslObj(domain()) << "\n";
  }

  if (withAccessOps) {
    if (withLabels)
      os << "- Access Operations\n";
    for (unsigned i = 0; i < accessOps.size(); ++i) {
      os << "    ";
      accessOps[i]->print(os, OpPrintingFlags());
      os << "\n";
      os << "    " << stringFromIslObj(accessRelations[i]) << "\n";
    }
  }

  if (withLabels)
    os << "- Operations\n";
  for (unsigned i = 0; i < allOps.size(); ++i) {
    os << "    ";
    allOps[i]->print(os, OpPrintingFlags());
    os << "\n";
  }
}

// DependenceGraph
// ----------------------------------------
void DependenceGraph::dump(llvm::raw_ostream &os) {
  os << "Dependence Graph\n"
     << "----------------\n";
  for (auto &node : nodes) {
    os << node->stmt->name << ":\n";
    for (auto &dep : node->dependents) {
      os << "  -> " << dep->stmt->name << "\n";
    }
  }
  os << "\n";

  os << "Dependency Graph\n"
     << "----------------\n";
  for (auto &node : nodes) {
    os << node->stmt->name << ":\n";
    for (auto &dep : node->dependencies) {
      os << "  -> " << dep->stmt->name << "\n";
    }
  }
  os << "\n";
}

int DependenceGraph::getNumDependencies() {
  int numDependencies = 0;
  for (auto &node : nodes)
    numDependencies += node->dependencies.size();
  return numDependencies;
}

void DependenceGraph::computeDependencies() {
  // Init dependencies with reverse dependences.
  for (auto &nodePtr : nodes) {
    Node *node = nodePtr.get();
    for (auto &succ : node->dependents) {
      bool exists = false;
      for (auto &pred : succ->dependencies) {
        if (pred == node) {
          exists = true;
          break;
        }
      }
      if (!exists && node != succ)
        succ->dependencies.push_back(node);
    }
  }

  // Propagate dependencies with a worklist algorithm.
  std::vector<Node*> worklist;
  for (auto &node : nodes)
    worklist.push_back(node.get());

  while (!worklist.empty()) {
    auto *node = worklist.back();
    worklist.pop_back();

    auto nodeDependencies = node->dependencies;
    for (auto *succ: nodeDependencies) {
      for (auto *pred : succ->dependencies) {
        bool exists = false;
        for (auto &dep : node->dependencies) {
          if (dep == pred) {
            exists = true;
            break;
          }
        }
        if (!exists && node != pred) {
          node->dependencies.push_back(pred);
          worklist.push_back(pred);
        }
      }
    }
  }
}

// Scop
// ----------------------------------------
Scop::Scop(Operation *op) : op(op) {
  ctx = isl_ctx_alloc();
  asmState = new AsmState(op);

  buildScopStmts();
  buildAccessRelationIslMaps();

  ScopStmtMap seenStmts;
  schedule = scheduleOperation(&seenStmts, op, *this, 0);
  computeFlowDependencies();
}

Scop::~Scop() {
  // TODO: Properly handle lifetime of all the isl objects, so that we don't get
  // a isl freeing warning when freeing the context.

  // isl_ctx_free(ctx);
  // delete asmState;
}

DependenceGraphPtr Scop::getDependenceGraph() {
  DependenceGraphPtr graph = std::make_shared<DependenceGraph>();

  std::unordered_map<ScopStmt *, DependenceGraph::Node*>
      stmtsToGraphNodes;

  // Create a node for each statement.
  for (auto &nameToStmt : namesToStmts) {
    ScopStmt &stmt = nameToStmt.second;
    DependenceGraph::NodePtr node =
        std::make_shared<DependenceGraph::Node>(&stmt);
    stmtsToGraphNodes[&stmt] = node.get();
    graph->nodes.push_back(node);
  }

  for (auto flowDep : flowDependencies.get_map_list()) {
    isl::map dep = flowDep.as_map();

    // Get the source and destination ids.
    isl::id srcId = dep.get_tuple_id(isl::dim::in);
    isl::id dstId = dep.get_tuple_id(isl::dim::out);

    // Get the source and destination names.
    std::string srcName = srcId.get_name();
    std::string dstName = dstId.get_name();

    // Get the source and destination statements.
    ScopStmt *srcStmt = lookupStmtByName(srcName);
    ScopStmt *dstStmt = lookupStmtByName(dstName);
    assert(srcStmt && dstStmt && "Statement not found");

    // Add the dependents to the graph.
    DependenceGraph::Node* srcNode = stmtsToGraphNodes[srcStmt];
    DependenceGraph::Node* dstNode = stmtsToGraphNodes[dstStmt];
    srcNode->dependents.push_back(dstNode);
  }

  graph->computeDependencies();

  return graph;
}

ScopStmt* Scop::lookupStmtByName(std::string name) {
  auto it = namesToStmts.find(name);
  if (it == namesToStmts.end())
    return nullptr;
  return &it->second;
}

ScopStmt *Scop::lookupStmtByOp(mlir::Operation *op) {
  for (auto &nameToStmt : namesToStmts) {
    ScopStmt &stmt = nameToStmt.second;

    for (auto &iop : stmt.accessOps) {
      if (op == iop) {
        return &stmt;
      }
    }
  }
  return nullptr;
}

llvm::SmallVector<ScopStmt> Scop::lookupStmtsByBlock(mlir::Block &block) {
  llvm::SmallVector<Operation *> ops;
  for (Operation &iop : block) {
    ops.push_back(&iop);
  }

  llvm::DenseMap<ScopStmt *, bool> stmtsMap;
  for (auto &nameToStmt : namesToStmts) {
    ScopStmt &stmt = nameToStmt.second;

    for (auto &op : stmt.accessOps) {
      for (auto &iop : ops) {
        if (op == iop) {
          stmtsMap[&stmt] = true;
        }
      }
    }
  }
  llvm::SmallVector<ScopStmt> result;
  for (auto stmt : stmtsMap) {
    result.push_back(*stmt.first);
  }

  return result;
}

void Scop::buildScopStmts() {
  llvm::SmallVector<Operation *> currentStmtAllOps;
  llvm::SmallVector<Operation *> currentStmtAccessOps;

  unsigned stmtIdx = 0;
  op->walk<WalkOrder::PreOrder>([&](Operation *op) {
    bool isSimpleOp = !isa<AffineForOp>(op) && !isa<AffineYieldOp>(op) &&
                      !isa<FunctionOpInterface>(op) && !isa<ModuleOp>(op) &&
                      !isa<func::ReturnOp>(op);
    bool isAccessOp = isa<AffineLoadOp>(op) || isa<AffineStoreOp>(op);
    bool isStoreOp = isa<AffineStoreOp>(op);

    if (isSimpleOp) {
      currentStmtAllOps.push_back(op);
    }

    if (isSimpleOp && isAccessOp) {
      currentStmtAccessOps.push_back(op);
    }

    if ((!isSimpleOp || isStoreOp)) {
      if (!currentStmtAccessOps.empty()) {
        std::string stmtName = "Stmt" + std::to_string(stmtIdx++);
        ScopStmt scopStmt(currentStmtAllOps, currentStmtAccessOps, stmtName);
        namesToStmts.insert({stmtName, scopStmt});
      }
      currentStmtAccessOps.clear();
      currentStmtAllOps.clear();
    }
  });

  if (!currentStmtAccessOps.empty()) {
    std::string stmtName = "Stmt" + std::to_string(stmtIdx++);
    ScopStmt scopStmt(currentStmtAllOps, currentStmtAccessOps, stmtName);
    namesToStmts.insert({stmtName, scopStmt});
  }
}

void Scop::buildAccessRelationIslMaps() {
  for (auto &nameToStmt : namesToStmts) {
    ScopStmt &stmt = nameToStmt.second;

    llvm::SmallVector<isl::map> ms;
    for (auto &op : stmt.accessOps) {
      // Build ISL access relation.
      isl::map m = getAccessRelationForOp(op, stmt.name);
      ms.push_back(m);
    }
    stmt.accessRelations = ms;
  }
}

void Scop::computeFlowDependencies() {
  isl::union_map reads;
  isl::union_map writes;
  for (auto &nameToStmt : namesToStmts) {
    ScopStmt &stmt = nameToStmt.second;

    for (unsigned i = 0; i < stmt.accessOps.size(); ++i) {
      Operation *op = stmt.accessOps[i];
      isl::union_map acc = stmt.accessRelations[i];
      if (isa<AffineLoadOp>(op)) {
        if (reads.is_null()) {
          reads = acc;
        } else {
          reads = reads.unite(acc);
        }
      } else if (isa<AffineStoreOp>(op)) {
        if (writes.is_null()) {
          writes = acc;
        } else {
          writes = writes.unite(acc);
        }
      }
    }
  }

  isl::union_access_info uai(reads);
  uai = uai.set_must_source(writes);
  uai = uai.set_schedule(schedule);
  isl::union_flow flow = uai.compute_flow();

  flowDependencies = flow.get_may_dependence();
}

isl::map Scop::getAccessRelationForOp(Operation *op, std::string &opName) {
  // Get the access relation using Presburger lib.
  MemRefAccess access(op);
  FlatAffineRelation rel;
  LogicalResult result = access.getAccessRelation(rel);
  assert(succeeded(result) && "Failed to get access relation");
  // dumpRelDetails(rel);

  // Create access relation space.
  isl::space accessRelSpace =
      isl::space(ctx, rel.getNumSymbolVars(), rel.getNumDomainDims(),
                 rel.getNumRangeDims());

  // Set tuple names.
  isl::id opId = isl::id::alloc(ctx, opName, op);
  isl::id memrefId;
  // std::string memrefName =
  //     "MemRef" +
  //     std::to_string(access.memref.cast<BlockArgument>().getArgNumber());
  std::string memrefName = "MemRef";
  if (auto loadOp = dyn_cast<AffineLoadOp>(op)) {
    memrefId = isl::id::alloc(ctx, memrefName, loadOp.getMemRef().getImpl());
  } else if (auto storeOp = dyn_cast<AffineStoreOp>(op)) {
    memrefId = isl::id::alloc(ctx, memrefName, storeOp.getMemref().getImpl());
  } else {
    llvm_unreachable("unexpected operation");
  }
  accessRelSpace =
      accessRelSpace.set_tuple_id(isl::dim::in, isl::id(ctx, opName));

  // Set input identifiers.
  for (unsigned i = 0; i < rel.getNumDomainDims(); ++i) {
    std::string asmId;
    raw_string_ostream so(asmId);
    rel.getValue(i).printAsOperand(so, *asmState);
    asmId = std::regex_replace(asmId, std::regex(R"(%)"), "");

    std::string id = asmId;
    // std::string id = "i" + std::to_string(i);

    isl::id islId = isl::id::alloc(ctx, id, rel.getValue(i).getImpl());
    accessRelSpace = accessRelSpace.set_dim_id(isl::dim::in, i, islId);
  }

  // Set param identifiers.
  for (unsigned i = 0; i < rel.getNumSymbolVars(); ++i) {
    // isl::id id =
    //     isl::id::alloc(ctx, "p_" + std::to_string(i),
    //                    rel.getValue(rel.getNumDimVars() + i).getImpl());
    accessRelSpace = accessRelSpace.set_dim_id(
        isl::dim::param, i, isl::id(ctx, "p_" + std::to_string(i)));
  }

  // Convert the access relation to isl::map.
  isl::basic_map accessRel = isl::basic_map::universe(accessRelSpace);
  unsigned localOffsetISL = rel.getNumRangeDims();
  accessRel = accessRel.insert_dims(isl::dim::out, localOffsetISL,
                                    rel.getNumLocalVars());

  // Copy inequality constraints.
  for (unsigned i = 0, e = rel.getNumInequalities(); i < e; ++i) {
    isl::constraint c =
        isl::constraint::alloc_inequality(accessRel.get_local_space());
    unsigned offset = 0;
    // Domain variables.
    for (unsigned j = 0, e = rel.getNumDomainDims(); j < e; ++j)
      c = c.set_coefficient_si(isl::dim::in, j, rel.atIneq64(i, offset + j));
    offset += rel.getNumDomainDims();
    // Range variables.
    for (unsigned j = 0, e = rel.getNumRangeDims(); j < e; ++j)
      c = c.set_coefficient_si(isl::dim::out, j, rel.atIneq64(i, offset + j));
    offset += rel.getNumRangeDims();
    // Symbol variables.
    for (unsigned j = 0, e = rel.getNumSymbolVars(); j < e; ++j)
      c = c.set_coefficient_si(isl::dim::param, j, rel.atIneq64(i, offset + j));
    offset += rel.getNumSymbolVars();
    // Local variables.
    for (unsigned j = 0, e = rel.getNumLocalVars(); j < e; ++j)
      c = c.set_coefficient_si(isl::dim::out, j + localOffsetISL,
                               rel.atIneq64(i, offset + j));

    c = c.set_constant_si(rel.atIneq64(i, rel.getNumCols() - 1));
    accessRel = accessRel.add_constraint(c);
  }

  // Copy equality constraints.
  for (unsigned i = 0, e = rel.getNumEqualities(); i < e; ++i) {
    isl::constraint c =
        isl::constraint::alloc_equality(accessRel.get_local_space());
    unsigned offset = 0;
    // Domain variables.
    for (unsigned j = 0, e = rel.getNumDomainDims(); j < e; ++j)
      c = c.set_coefficient_si(isl::dim::in, j, rel.atEq64(i, offset + j));
    offset += rel.getNumDomainDims();
    // Range variables.
    for (unsigned j = 0, e = rel.getNumRangeDims(); j < e; ++j)
      c = c.set_coefficient_si(isl::dim::out, j, rel.atEq64(i, offset + j));
    offset += rel.getNumRangeDims();
    // Symbol variables.
    for (unsigned j = 0, e = rel.getNumSymbolVars(); j < e; ++j)
      c = c.set_coefficient_si(isl::dim::param, j, rel.atEq64(i, offset + j));
    offset += rel.getNumSymbolVars();
    // Local variables.
    for (unsigned j = 0, e = rel.getNumLocalVars(); j < e; ++j)
      c = c.set_coefficient_si(isl::dim::out, j + localOffsetISL,
                               rel.atEq64(i, offset + j));

    c = c.set_constant_si(rel.atEq64(i, rel.getNumCols() - 1));
    accessRel = accessRel.add_constraint(c);
  }

  isl::map accessRelM = isl::map(accessRel);
  accessRelM = accessRelM.project_out(isl::dim::out, localOffsetISL,
                                      rel.getNumLocalVars());
  accessRelM = accessRelM.set_tuple_id(isl::dim::out, memrefId);

  return accessRelM;
}

void Scop::dump(raw_ostream &os) {
  for (auto &nameToStmt : namesToStmts) {
    ScopStmt &stmt = nameToStmt.second;

    stmt.dump(os);
    os << "\n";
  }

  os << "Schedule\n";
  os << "    " << stringFromIslObj(schedule) << "\n\n";
  os << "    " << stringFromIslObj(schedule.get_map()) << "\n\n";
  dumpIslObj(schedule.get_root(), os);

  os << "FlowDependences\n";
  for (auto m : flowDependencies.get_map_list()) {
    os << "    " << stringFromIslObj(m) << "\n";
  }

  os << "\n";
}

void Scop::toDot(raw_ostream &os) {
  os << "digraph {\n";

  // Edges.
  for (auto um : flowDependencies.get_map_list()) {
    isl::map m = um.as_map();

    os << "  " << stringFromIslObj(m.get_tuple_id(isl::dim::in)) << " -> "
       << stringFromIslObj(m.get_tuple_id(isl::dim::out));

    // m = m.set_tuple_id(isl::dim::in, isl::id(ctx, "s"));
    // m = m.set_tuple_id(isl::dim::out, isl::id(ctx, "s"));

    std::string label = "";
    label += stringFromIslObj(m) + "\\l";

    os << " [label=\"" << label << "\"];\n";
  }

  // Nodes.
  for (auto &nameToStmt : namesToStmts) {
    ScopStmt &stmt = nameToStmt.second;

    os << "  " << stmt.name;
    os << " [shape=box, ";

    os << "label=\"";
    std::string stmtStr;
    raw_string_ostream rso(stmtStr);
    stmt.dump(rso, false, false, false, false);

    // Remove type annotations
    stmtStr = std::regex_replace(stmtStr, std::regex(R"( :.*\n)"), "\n");
    // Remove leading whitespaces before instructions
    stmtStr = std::regex_replace(stmtStr, std::regex(R"(\n[\s]*)"), "\n");
    stmtStr = std::regex_replace(stmtStr, std::regex(R"(^[\s]*)"), "");

    // Align left in dot graph
    stmtStr = std::regex_replace(stmtStr, std::regex(R"(\n)"), "\\l");
    os << stmtStr;
    os << "\"];\n";
  }

  os << "}\n";
}

void Scop::toDotStmts(raw_ostream &os) {
  os << "digraph {\n";

  mlir::DenseMap<mlir::Operation *, std::string> opStrs;
  for (auto &nameToStmt : namesToStmts) {
    ScopStmt &stmt = nameToStmt.second;

    os << "subgraph cluster_" + stmt.name << " {\n";
    os << "  label = \"" << stmt.name << "\";\n";

    // Nodes.
    for (auto &op : stmt.allOps) {
      if (opStrs.find(op) == opStrs.end()) {
        std::string opStr = "n" + std::to_string(opStrs.size());
        opStrs[op] = opStr;

        os << "  " << opStr;
        os << " [shape=box, ";
        os << "label=\"" << op->getName().getStringRef().str() << "\"";
        os << "];\n";
      }
    }

    // Edges.
    for (auto &op : stmt.allOps) {
      // Definitions from ops
      for (auto operand : op->getOperands()) {
        if (auto *defOp = operand.getDefiningOp()) {
          if (opStrs.find(defOp) != opStrs.end()) {
            os << "  " << opStrs[defOp] << " -> " << opStrs[op] << ";\n";
          }
        }
      }
    }

    os << "}\n";
  }

  // External vars or args
  mlir::DenseMap<mlir::Value, std::string> undefArgs;
  mlir::DenseMap<mlir::Operation *, std::string> undefOps;
  os << "subgraph cluster_external {\n";
  os << "  label = \"External Vars and Args\";\n";
  for (auto &nameToStmt : namesToStmts) {
    ScopStmt &stmt = nameToStmt.second;

    for (auto &op : stmt.allOps) {
      for (auto operand : op->getOperands()) {
        mlir::Operation *iop = operand.getDefiningOp();
        // It's a variable
        if (iop) {
          if (opStrs.find(iop) == opStrs.end()) {
            // Create node
            if (undefOps.find(iop) == undefOps.end()) {
              std::string undefStr =
                  "var" + std::to_string(undefArgs.size() + undefOps.size());
              undefOps[iop] = undefStr;

              os << "  " << undefStr;
              os << " [shape=box, ";
              os << "label=\"Ext\"";
              os << "];\n";
            }
            // Create edge
            os << "  " << undefOps[iop] << " -> " << opStrs[op] << ";\n";
          }
          // It's an argument
        } else {
          auto blockArg = operand.cast<BlockArgument>();

          // Create node
          if (undefArgs.find(blockArg) == undefArgs.end()) {
            std::string undefStr =
                "var" + std::to_string(undefArgs.size() + undefOps.size());
            undefArgs[blockArg] = undefStr;

            os << "  " << undefStr;
            os << " [shape=box, ";
            os << "label=\"Ext\"";
            os << "];\n";
          }
          // Create edge
          os << "  " << undefArgs[blockArg] << " -> " << opStrs[op] << ";\n";
        }
      }
    }
  }
  os << "}\n";

  os << "}\n";
}

void dumpRelDetails(FlatAffineRelation rel) {
  llvm::outs() << "--------Relation--------\n";

  rel.dump();
  for (unsigned int i = 0; i < rel.getNumVars(); ++i) {
    llvm::outs() << "Var " << i << "\t";

    std::string varKind;
    switch (rel.getVarKindAt(i)) {
    case mlir::presburger::VarKind::Symbol:
      varKind = "Symbol";
      break;
    case mlir::presburger::VarKind::Local:
      varKind = "Local";
      break;
    case mlir::presburger::VarKind::Domain:
      varKind = "Domain";
      break;
    case mlir::presburger::VarKind::Range:
      varKind = "Range";
      break;
    }
    llvm::outs() << varKind << "\t";

    if (rel.hasValue(i)) {
      rel.getValue(i).dump();
    } else {
      llvm::outs() << "\n";
    }
  }

  llvm::outs() << "Dims: " << rel.getNumDomainDims() << " "
               << rel.getNumRangeDims() << "\n";
}

BoostGraph constructBoostGraph(DependenceGraphPtr& dg) {
  BoostGraph g;
  std::unordered_map<ScopStmt *, BoostGraph::vertex_descriptor> vertexMap;

  // Add nodes.
  for (auto &node : dg->nodes) {
    vertexMap[node->stmt] = boost::add_vertex(g);
    g[vertexMap[node->stmt]].stmt = node->stmt;
  }

  // Add edges.
  for (auto &node : dg->nodes) {
    for (auto &dep : node->dependents) {
      auto *src = node->stmt;
      auto *dst = dep->stmt;

      auto srcVertex = vertexMap[src];
      auto dstVertex = vertexMap[dst];

      boost::add_edge(srcVertex, dstVertex, g);
    }
  }

  return g;
}

void printBoostGraph(BoostGraph &g) {
  // Print the graph
  for (auto v : boost::make_iterator_range(boost::vertices(g))) {
    llvm::outs() << "Vertex " << v << " has edges to: ";
    for (auto e : boost::make_iterator_range(boost::out_edges(v, g))) {
      llvm::outs() << boost::target(e, g) << " ";
    }
    llvm::outs() << "\n";
  }
}

namespace boost {
void renumber_vertex_indices(BoostGraph const &) {}
} // namespace boost

struct CycleRecorder {
  template <typename Path, typename BoostGraph>
  void cycle(const Path &p, const BoostGraph &g) {
    numCycles++;
  }
  int numCycles = 0;
};

int computeNumCycles(BoostGraph const &g) {
  CycleRecorder cycleRecorder;
  boost::tiernan_all_cycles(g, cycleRecorder);
  return cycleRecorder.numCycles;
}
