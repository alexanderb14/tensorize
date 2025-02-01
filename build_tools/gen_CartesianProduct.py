import argparse

header = """/*===- Cartesian Product Generators -----------------------------*- C++ -*-===*\
|*                                                                            *|
|* The arguments to ranges::views::cartesian_product need to be known         *|
|* at compile time. Therefore, this generator-approach. TODO: Replace.        *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

#include "CartesianProduct.h"

#include "synthesis/Generators.h"

#include <range/v3/all.hpp>
#include <range/v3/view/cartesian_product.hpp>

#include <iostream>
#include <vector>

using namespace llvm;
using namespace mlir;

int getTotalNumOps(ArgTuple tuple) {
  int totalNumOps = 1;
  for (const auto &operand : tuple.operands) {
    totalNumOps += operand->getNumOps();
  }
  return totalNumOps;
}

"""

body_header = """
std::vector<ArgTuple>
CartesianProduct::generate(std::vector<std::vector<CandidatePtr>> &operands,
                           std::vector<std::vector<Attribute>> &attributes,
                           std::vector<std::vector<RegionPtr>> &regions) {
  unsigned numOperands = operands.size();
  unsigned numAttributes = attributes.size();
  unsigned numRegions = regions.size();

  std::vector<ArgTuple> ret;
"""

body_footer = """
  llvm::outs() << "Unsupported number of operands (" << numOperands
               << "), attributes (" << numAttributes << "), regions ("
               << numRegions << ")\\n";
  assert(false);
}
"""


def get_condition(numOperands, numAttributes, numRegions):
    return "numOperands == {} && numAttributes == {} && numRegions == {}".format(numOperands, numAttributes, numRegions)


def get_action(numOperands, numAttributes, numRegions):
    args = []
    for i in range(numOperands):
        args += ["operands[{}]".format(i)]
    for i in range(numAttributes):
        args += ["attributes[{}]".format(i)]
    for i in range(numRegions):
        args += ["regions[{}]".format(i)]

    src = "    auto cands = ranges::views::cartesian_product(" + ", ".join(
        args) + ");\n"

    src += "    for (auto cand : cands) {\n"
    src += "      ArgTuple tuple;\n"

    counter = 0

    operands = []
    for i in range(numOperands):
        operands += ["std::get<{}>(cand)".format(counter)]
        counter += 1
    src += "      tuple.operands = {" + ", ".join(operands) + "};\n"

    attributes = []
    for i in range(numAttributes):
        attributes += ["std::get<{}>(cand)".format(counter)]
        counter += 1
    src += "      tuple.attributes = {" + ", ".join(attributes) + "};\n"

    regions = []
    for i in range(numRegions):
        regions += ["std::get<{}>(cand)".format(counter)]
        counter += 1
    src += "      tuple.regions = {" + ", ".join(regions) + "};\n"

    src += "      if (getTotalNumOps(tuple) <= maxNumOps)\n"
    src += "        ret.push_back(tuple);\n"
    src += "    }\n"
    src += "    return ret;\n"

    return src


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_operands", type=int)
    parser.add_argument("--max_attributes", type=int)
    parser.add_argument("--max_regions", type=int)
    parser.add_argument("--output", type=str)
    args = parser.parse_args()

    src = header
    src += body_header
    for i in range(0, args.max_operands + 1):
        for j in range(0, args.max_attributes + 1):
            for k in range(0, args.max_regions + 1):
                if i == 0 and j == 0 and k == 0:
                    continue
                src += "  if (" + get_condition(i, j, k) + ") {\n"
                src += get_action(i, j, k)
                src += "  }\n"
    src += body_footer

    with open(args.output, "w") as f:
        f.write(src)


if __name__ == "__main__":
    main()
