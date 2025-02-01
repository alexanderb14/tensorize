#ifndef IRSYNTH_PASSES_TD
#define IRSYNTH_PASSES_TD

include "mlir/Pass/PassBase.td"

def AnnotateLastStoredMemrefArg : Pass<"annotate-output-arg", "mlir::ModuleOp"> {
  let summary = "Annotation of the last stored memref argument";
  let description = [{
  }];
  let constructor = "createCleanupPass()";
}

def ChangeSizes: Pass<"change-sizes", "mlir::ModuleOp"> {
  let summary = "Minifies memrefs according to a prime number sequence.";
  let description = [{
  }];
  let constructor = "createChangeSizesPass()";
  let options = [
   Option<"mode", "mode", "std::string",
           /*default=*/"\"minify\"", "Allowed options: minify, restore">,
   Option<"sizeMode", "sizes", "std::string",
           /*default=*/"\"Primes\"", "Allowed options: Primes, Uniform">,
  ];
}

def Cleanup: Pass<"cleanup", "mlir::ModuleOp"> {
  let summary = "Cleans up the IR after the synthesis";
  let description = [{
  }];
  let constructor = "createCleanupPass()";
}

def CopyModifiedMemrefs: Pass<"copy-modified-memrefs", "mlir::ModuleOp"> {
  let summary = "Creates copies of stored memrefs and replaces their original uses "
                "with them";
  let description = [{
  }];
  let constructor = "createCopyModifiedMemrefsPass()";
}

def FoldToTensorToMemrefPair: Pass<"fold-totensor-tomemref-pairs", "mlir::ModuleOp"> {
  let summary = "Folds to_tensor and to_memref pairs";
  let description = [{
  }];
  let constructor = "createFoldToTensorToMemrefPairPass()";
}

def LoopDistribution: Pass<"distribute-loops", "mlir::ModuleOp"> {
  let summary = "Distributes loop";
  let description = [{
  }];
  let constructor = "createLoopDistributionPass()";
}

def LoopOutline: Pass<"outline-loops", "mlir::ModuleOp"> {
  let summary = "Outlines all top-level loops into seperate functions and calls "
                "them";
  let description = [{
  }];
  let constructor = "createLoopOutlinePass()";
}

def MemrefToScf: Pass<"memref-to-scf", "mlir::ModuleOp"> {
  let summary = "Convert memref ops to scf ops";
  let description = [{
  }];
  let constructor = "createMemrefToScfPass()";
}

def MemrefRank0ToScalar: Pass<"memref-rank0-to-scalar", "mlir::ModuleOp"> {
  let summary = "Convert memref of rank 0 to scalars";
  let description = [{
  }];
  let constructor = "createMemrefRank0ToScalarPass()";
}

def PrepareTarget: Pass<"prepare-target", "mlir::ModuleOp"> {
  let summary = "Prepares the function that has an irsynth.target argument "
                "for compilation with XLA.";
  let description = [{
  }];
  let constructor = "createPrepareTargetPass()";
}

def TargetOutline: Pass<"outline-target", "mlir::ModuleOp"> {
  let summary = "Outlines the target program into a seperate function and calls "
                "it.";
  let description = [{
  }];
  let constructor = "createTargetOutlinePass()";
}

#endif //IRSYNTH_PASSES_TD