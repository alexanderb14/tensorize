#include "Spec.h"

void Spec::dumpAsPython() {
  llvm::outs() << "inputs = {\n";
  printArgs(inputs, llvm::outs());
  llvm::outs() << "}\n";
  llvm::outs() << "output = ";
  printArgs({output}, llvm::outs());
}
