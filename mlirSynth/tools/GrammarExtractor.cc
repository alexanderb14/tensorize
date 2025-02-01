#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/TableGen/AttrOrTypeDef.h"
#include "mlir/TableGen/Attribute.h"
#include "mlir/TableGen/GenInfo.h"
#include "mlir/TableGen/Operator.h"
#include "mlir/Tools/mlir-tblgen/MlirTblgenMain.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/TableGen/Main.h"
#include "llvm/TableGen/Record.h"
#include "llvm/TableGen/TableGenBackend.h"

#include <map>
#include <regex>
#include <set>
#include <string>

using namespace llvm;
using namespace mlir;
using namespace mlir::tblgen;

// Utility functions
// -----------------------------------------------------------------------------
std::string replaceSpecialChars(const std::string &str,
                                const std::string &replaceWith = "") {
  return std::regex_replace(str, std::regex("[^a-zA-Z0-9]"), replaceWith);
}

std::string cutAndUppercaseFirstChar(std::string qualType) {
  auto pos = qualType.rfind("::");
  if (pos != std::string::npos)
    return qualType.substr(pos + 2);
  return qualType;
}

std::string getAttrGenFnName(AttrOrTypeDef &attrDef) {
  std::string dialectName = attrDef.getDialect().getName().str();
  dialectName[0] = toupper(dialectName[0]);

  std::string attrName = attrDef.getCppClassName().str();
  return "gen" + dialectName + attrName;
}

std::string getAttrGenFnName(tblgen::Attribute &attr) {
  std::string dialectName;
  if (attr.getDialect()) {
    dialectName = attr.getDialect().getName().str();
    dialectName[0] = toupper(dialectName[0]);
  }

  std::string attrName = attr.getStorageType().str();
  return "gen" + dialectName + cutAndUppercaseFirstChar(attrName);
}

std::string getAttrGenFnName(std::string attrName) {
  attrName = replaceSpecialChars(attrName);
  attrName[0] = toupper(attrName[0]);
  return "gen" + attrName;
}

std::vector<Record *> getOpDefinitions(const RecordKeeper &recordKeeper) {
  if (!recordKeeper.getClass("Op"))
    return {};
  return recordKeeper.getAllDerivedDefinitions("Op");
}

std::vector<Record *> getTypeDefinitions(const RecordKeeper &recordKeeper) {
  if (!recordKeeper.getClass("TypeDef"))
    return {};
  return recordKeeper.getAllDerivedDefinitions("TypeDef");
}

std::vector<Record *> getAttrDefinitions(const RecordKeeper &recordKeeper) {
  if (!recordKeeper.getClass("AttrDef"))
    return {};
  return recordKeeper.getAllDerivedDefinitions("AttrDef");
}

std::vector<std::string> getUsedOpAndResTypes(const RecordKeeper &records) {
  std::set<std::string> types;
  for (auto *record : getOpDefinitions(records)) {
    auto tblgenOp = Operator(record);
    for (auto &operand : tblgenOp.getOperands()) {
      types.insert(operand.constraint.getDefName().str());
    }
    for (auto &result : tblgenOp.getResults()) {
      types.insert(result.constraint.getDefName().str());
    }
  }

  return std::vector<std::string>(types.begin(), types.end());
}

std::set<std::string> getAllAttrGenFnNames(const RecordKeeper &records) {
  std::set<std::string> genFns;
  for (auto *record : getOpDefinitions(records)) {
    auto tblgenOp = Operator(record);
    for (int i = 0; i < tblgenOp.getNumAttributes(); ++i) {
      auto &attr = tblgenOp.getAttribute(i);
      genFns.insert(getAttrGenFnName(attr.attr));
    }
  }

  return genFns;
}

std::set<std::string> getAttrDefGenFnNames(const RecordKeeper &records) {
  std::set<std::string> genFns;

  auto attrDefs = records.getAllDerivedDefinitionsIfDefined("AttrDef");
  for (auto *attrDefRecord : attrDefs) {
    AttrOrTypeDef attrDef(attrDefRecord);
    genFns.insert(getAttrGenFnName(attrDef));
  }

  return genFns;
}

std::set<std::string> getAttrNonDefGenFnNames(const RecordKeeper &records) {
  auto attrDefGenFns = getAttrDefGenFnNames(records);
  auto allGenFns = getAllAttrGenFnNames(records);

  // Compute set difference attrGenFns - attrDefGenFns
  std::set<std::string> attrGenFnsOnly;
  std::set_difference(allGenFns.begin(), allGenFns.end(), attrDefGenFns.begin(),
                      attrDefGenFns.end(),
                      std::inserter(attrGenFnsOnly, attrGenFnsOnly.begin()));

  return attrGenFnsOnly;
}

// Include guard emitters
// -----------------------------------------------------------------------------
void emitIncludeGuardStart(raw_ostream &os, const std::string &guard) {
  os << "#ifndef " << guard << "\n";
  os << "#define " << guard << "\n";
  os << "\n";
}

void emitIncludeGuardEnd(raw_ostream &os, const std::string &guard) {
  os << "#endif // " << guard << "\n";
}

// Header emitters
// -----------------------------------------------------------------------------
void emitHdrIncludes(raw_ostream &os) {
  os << R"(
#include "mlir/IR/Attributes.h"
#include "mlir/IR/MLIRContext.h"

#include <memory>
#include <string>
#include <vector>

)";
}

void emitSrcIncludes(raw_ostream &os) {
  os << R"(
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

)";
}

// Namespace emitters
// -----------------------------------------------------------------------------
void emitNamespaceStart(raw_ostream &os, const std::string &ns) {
  os << "namespace " << ns << " {\n";
}

void emitNamespaceEnd(raw_ostream &os, const std::string &ns) {
  os << "} // namespace " << ns << "\n";
}

// Op / Attr type emitters
// -----------------------------------------------------------------------------
void emitUsedOpAndResTypesAsEnum(const RecordKeeper &records, raw_ostream &os) {
  auto types = getUsedOpAndResTypes(records);

  os << "enum OpAndResType {\n";
  unsigned size = types.size();
  for (auto &type : types) {
    os << "  " << type;
    if (--size > 0) {
      os << ",\n";
    } else {
      os << "\n";
    }
  }
  os << "};\n";
}

void emitUsedAttrTypeValues(const RecordKeeper &records, raw_ostream &os) {
  auto enumDefs = records.getAllDerivedDefinitionsIfDefined("EnumAttrInfo");
  for (auto *enumDefRecord : enumDefs) {
    EnumAttr enumAttr(enumDefRecord);
    os << enumAttr.getEnumClassName() << ", " << enumAttr.getCppNamespace()
       << "\n";
    for (auto enumerant : enumAttr.getAllCases()) {
      os << "  " << enumerant.getSymbol() << "\n";
    }
  }
  os << "\n";

  auto attrDefs = records.getAllDerivedDefinitionsIfDefined("AttrDef");
  for (auto *attrDefRecord : attrDefs) {
    AttrOrTypeDef attrDef(attrDefRecord);
    os << attrDef.getCppClassName() << "\n";

    for (auto param : attrDef.getParameters()) {
      os << "  " << param.getName();
      os << " : ";

      std::string paramType = param.getCppType().str();
      if (auto *paramDefInit = dyn_cast<llvm::DefInit>(param.getDef())) {
        auto *rec = paramDefInit->getDef();
        if (!rec->isSubClassOf("EnumParameter"))
          paramType = rec->getName();
      }

      // Exctract all what is after the rightmost ::
      paramType = cutAndUppercaseFirstChar(paramType);
      os << paramType << "\n";
    }
  }
}

// Enum type generator emitters
// -----------------------------------------------------------------------------
std::unordered_map<std::string, Record *>
getEnumAttrDefs(const RecordKeeper &records) {
  std::unordered_map<std::string, Record *> enumAttrDefs;
  auto enumDefs = records.getAllDerivedDefinitionsIfDefined("EnumAttrInfo");
  for (auto *enumDefRecord : enumDefs) {
    EnumAttr enumAttr(enumDefRecord);
    std::string enumName = enumAttr.getCppNamespace().str() +
                           "::" + enumAttr.getEnumClassName().str();
    enumAttrDefs[enumName] = enumDefRecord;
  }
  return enumAttrDefs;
}

void emitEnumerants(std::unordered_map<std::string, Record *> &enumAttrDefs,
                    AttrOrTypeParameter &param, raw_ostream &os) {
  std::string enumQualType = param.getCppStorageType().str();
  auto enumDefRecord = enumAttrDefs.find(enumQualType);
  if (enumDefRecord != enumAttrDefs.end()) {
    std::string enumerantsStr;

    EnumAttr enumAttr(enumDefRecord->second);
    for (auto enumerant : enumAttr.getAllCases()) {
      std::string enumerantQualType =
          enumQualType + "::" + enumerant.getSymbol().str();
      enumerantsStr += "    " + enumerantQualType + ",\n";
    }

    os << "  std::vector<" + enumQualType + "> " + param.getName() +
              "Enumerants = {\n";
    os << enumerantsStr;
    os << "  };\n";
  } else {
    os << "  std::vector<" + enumQualType + "> " + param.getName() +
              "Enumerants = " + getAttrGenFnName(enumQualType) + "(attributeName);\n";
  }
}

std::set<std::string> getTypesUsedInAttrDefs(const RecordKeeper &records) {
  std::set<std::string> enumerandTypes;

  auto enumAttrDefs = getEnumAttrDefs(records);
  auto attrDefs = records.getAllDerivedDefinitionsIfDefined("AttrDef");
  for (auto *attrDefRecord : attrDefs) {
    AttrOrTypeDef attrDef(attrDefRecord);

    for (auto param : attrDef.getParameters()) {
      std::string enumQualType = param.getCppStorageType().str();

      // Only add defs that are in attr defs
      auto enumDefRecord = enumAttrDefs.find(enumQualType);
      if (enumDefRecord == enumAttrDefs.end()) {
        // Remove all special characters

        enumerandTypes.insert(enumQualType);
      }
    }
  }

  return enumerandTypes;
}

void emitAttrGenFns(const RecordKeeper &records, raw_ostream &os) {
  // AttrDef
  auto enumAttrDefs = getEnumAttrDefs(records);
  auto attrDefs = records.getAllDerivedDefinitionsIfDefined("AttrDef");
  for (auto *attrDefRecord : attrDefs) {
    AttrOrTypeDef attrDef(attrDefRecord);

    // Emit function declaration.
    os << "std::vector<mlir::Attribute> "
       << "AttributeGeneratorBase::" << getAttrGenFnName(attrDef) << "(const std::string attributeName) {\n";

    // Emit enumerants.
    for (auto param : attrDef.getParameters()) {
      emitEnumerants(enumAttrDefs, param, os);
    }

    // Emit attribute generation.
    os << "  std::vector<mlir::Attribute> ret;\n";

    // - For loop header.
    int paramIdx = 0;
    for (auto param : attrDef.getParameters()) {
      std::string enumerantName = "v" + std::to_string(paramIdx);
      os << std::string(2 + paramIdx * 2, ' ')
         << "for (const auto &" + enumerantName + " : " + param.getName() +
                "Enumerants) {\n";
      paramIdx++;
    }

    // - Call.
    os << std::string(2 + paramIdx * 2, ' ');
    os << "ret.push_back(::mlir::" << attrDef.getDialect().getName()
       << "::" << attrDef.getCppClassName().str() << "::get(&ctx, \n";

    int paramIdx2 = 0;
    for (auto param : attrDef.getParameters()) {
      (void)param; // Suppress unused variable warning.

      std::string enumerantName = "v" + std::to_string(paramIdx2);
      os << std::string(4 + paramIdx * 2, ' ');
      os << enumerantName;

      if (paramIdx2 < paramIdx - 1)
        os << ",\n";
      paramIdx2++;
    }

    os << "));\n";

    // - For loop footer.
    for (int i = paramIdx; i > 0; i--) {
      os << std::string(i * 2, ' ') << "}\n";
    }
    os << "  return ret;\n";

    os << "}\n\n";
  }

  std::string notImplementedWarning = "WARNING: Not implemented";

  // Attr
  auto attrNonDefsGenFnNames = getAttrNonDefGenFnNames(records);
  for (auto &fnName : attrNonDefsGenFnNames) {
    os << "std::vector<mlir::Attribute> "
       << "AttributeGeneratorBase::" << fnName << "(const std::string attributeName) {\n";
    os << "  llvm::outs() << \"" << notImplementedWarning << ": " << fnName
       << "\\n\";\n";
    os << "  std::vector<mlir::Attribute> ret;\n";
    os << "  return ret;\n";
    os << "}\n\n";
  }

  auto typesUsedInAttrDefs = getTypesUsedInAttrDefs(records);
  for (auto &type : typesUsedInAttrDefs) {
    std::string fnName = getAttrGenFnName(type);

    os << "std::vector<" << type << "> "
       << "AttributeGeneratorBase::" << fnName << "(const std::string attributeName) {\n";
    os << "  llvm::outs() << \"" << notImplementedWarning << ": " << fnName
       << "\\n\";\n";
    os << "  std::vector<" << type << "> ret;\n";
    os << "  return ret;\n";
    os << "}\n\n";
  }
}

void emitAttrGenClass(const RecordKeeper &records, raw_ostream &os) {
  auto enumAttrDefs = getEnumAttrDefs(records);

  os << R"(
class AttributeGeneratorBase {
public:
  AttributeGeneratorBase(mlir::MLIRContext &ctx) : ctx(ctx) {}
  virtual ~AttributeGeneratorBase() = default;
)";

  auto attrDefGenFnNames = getAttrDefGenFnNames(records);
  os << "\n";
  os << "  // AttrDef generators. (exhaustively synthesize all combinations)\n";
  for (auto &fnName : attrDefGenFnNames) {
    os << "  std::vector<mlir::Attribute> " << fnName << "(const std::string attributeName);\n";
  }

  auto attrNonDefsGenFnNames = getAttrNonDefGenFnNames(records);
  os << "\n";
  os << "  // Attr generators. (to be derived and implemented)\n";
  for (auto &fnName : attrNonDefsGenFnNames) {
    os << "  virtual std::vector<mlir::Attribute> " << fnName << "(const std::string attributeName);\n";
  }

  auto typesUsedInAttrDefs = getTypesUsedInAttrDefs(records);
  os << "\n";
  os << "  // Types used in enums. (to be derived and implemented)\n";
  for (auto &typeName : typesUsedInAttrDefs) {
    os << "  virtual std::vector<" << typeName << "> "
       << getAttrGenFnName(typeName) << "(const std::string attributeName);\n";
  }

  os << R"(
protected:
  mlir::MLIRContext &ctx;
};
using AttributeGeneratorBasePtr = std::shared_ptr<AttributeGeneratorBase>;
)";
}

// Grammar Operation emitters
// -----------------------------------------------------------------------------
void emitAbstractOpClass(raw_ostream &os) {
  os << R"(
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
)";
}

void emitConcreteOps(const RecordKeeper &records, raw_ostream &os) {
  for (auto *record : getOpDefinitions(records)) {
    auto tblgenOp = Operator(record);

    std::string opName = replaceSpecialChars(tblgenOp.getOperationName(), "_");
    os << "class " << opName << " : public GrammarOp {\n";
    os << "public:\n";

    os << "  unsigned getNumOperands() const override {";
    os << " return " << tblgenOp.getNumOperands() << ";";
    os << " }\n";

    os << "  unsigned getNumAttributes() const override {";
    os << " return " << tblgenOp.getNumAttributes() << ";";
    os << " }\n";

    os << "  unsigned getNumRegions() const override {";
    os << " return " << tblgenOp.getNumRegions() << ";";
    os << " }\n";

    os << "  unsigned getNumResults() const override {";
    os << " return " << tblgenOp.getNumResults() << ";";
    os << " }\n";

    // Operands
    os << "  OpAndResType getOperandType(unsigned index) const override {\n";
    os << "    switch (index) {\n";
    for (int i = 0; i < tblgenOp.getNumOperands(); ++i) {
      auto &operand = tblgenOp.getOperand(i);
      os << "      case " << i << ": return " << operand.constraint.getDefName()
         << ";\n";
    }
    os << "    }\n";
    os << "    assert(false && \"Invalid operand index\");\n";
    os << "  }\n";

    os << "  std::string getOperandName(unsigned index) const override {\n";
    os << "    switch (index) {\n";
    for (int i = 0; i < tblgenOp.getNumOperands(); ++i) {
      auto &operand = tblgenOp.getOperand(i);
      os << "      case " << i << ": return \"" << operand.name << "\";\n";
    }
    os << "    }\n";
    os << "    assert(false && \"Invalid operand index\");\n";
    os << "  }\n";

    // Attributes
    os << "  mlir::Attribute getAttributeType(unsigned index) const override "
          "{\n";
    os << "    switch (index) {\n";
    for (int i = 0; i < tblgenOp.getNumAttributes(); ++i) {
      auto &attr = tblgenOp.getAttribute(i);
      os << "      case " << i << ": return " << attr.attr.getStorageType()
         << "()"
         << ";\n";
    }
    os << "    }\n";
    os << "    assert(false && \"Invalid attribute index\");\n";
    os << "  }\n";

    os << "  std::string getAttributeName(unsigned index) const override {\n";
    os << "    switch (index) {\n";
    for (int i = 0; i < tblgenOp.getNumAttributes(); ++i) {
      auto &attr = tblgenOp.getAttribute(i);
      os << "      case " << i << ": return \"" << attr.name.str() << "\";\n";
    }
    os << "    }\n";
    os << "    assert(false && \"Invalid attribute index\");\n";
    os << "  }\n";

    os << "  bool isAttributeRequired(unsigned index) const override {\n";
    os << "    switch (index) {\n";
    for (int i = 0; i < tblgenOp.getNumAttributes(); ++i) {
      auto &attr = tblgenOp.getAttribute(i);
      std::string isRequired = attr.attr.isOptional() ? "false" : "true";
      os << "      case " << i << ": return " << isRequired << ";\n";
    }
    os << "    }\n";
    os << "    assert(false && \"Invalid attribute index\");\n";
    os << "  }\n";

    os << "  std::vector<std::vector<mlir::Attribute>> "
          "genAttributes(AttributeGeneratorBasePtr attrGen) const override {\n";
    os << "    std::vector<std::vector<mlir::Attribute>> attrs;\n";
    for (int i = 0; i < tblgenOp.getNumAttributes(); ++i) {
      auto &attr = tblgenOp.getAttribute(i);

      auto genFnName = getAttrGenFnName(attr.attr);
      auto attrName = attr.name.str();

      std::string prefix = "";
      if (attr.attr.isOptional())
        prefix = "// ";
      os << "    " + prefix + "attrs.push_back(attrGen->" << genFnName
         << "(\"" + attrName + "\"));\n";
    }
    os << "    return attrs;\n";
    os << "  }\n";

    // Results
    os << "  OpAndResType getResultType(unsigned index) const override {\n";
    os << "    switch (index) {\n";
    for (int i = 0; i < tblgenOp.getNumResults(); ++i) {
      auto &result = tblgenOp.getResult(i);
      os << "      case " << i << ": return " << result.constraint.getDefName()
         << ";\n";
    }
    os << "    }\n";
    os << "    assert(false && \"Invalid result index\");\n";
    os << "  }\n";

    os << "};\n";
    os << "\n";
  }
}

void emitOpAndResTypeToStringDecl(raw_ostream &os) {
  os << "std::string opAndResTypeToString(OpAndResType type);\n";
}

void emitOpAndResTypeToStringFn(const RecordKeeper &records, raw_ostream &os) {
  auto types = getUsedOpAndResTypes(records);

  os << "std::string opAndResTypeToString(OpAndResType type) {\n";
  for (auto &type : types) {
    os << "  if (type == " << type << ") return \"" << type << "\";\n";
  }
  os << "  assert(false && \"Invalid OpAndResType\");\n";
  os << "}\n";
  os << "\n";
}

void emitConstructorDecl(raw_ostream &os) {
  os << "GrammarOpPtr createGrammarOp(std::string name);\n";
  os << "\n";
}

void emitConstructorFn(const RecordKeeper &records, raw_ostream &os) {
  os << "GrammarOpPtr createGrammarOp(std::string name) {\n";
  for (auto *record : getOpDefinitions(records)) {
    auto tblgenOp = Operator(record);

    std::string opName = replaceSpecialChars(tblgenOp.getOperationName(), "_");

    os << "  if (name == \"" << tblgenOp.getOperationName() << "\")\n";
    os << "    return std::make_unique<" << opName << ">();\n";
  }
  os << "  assert(false && \"Invalid op name\");\n";
  os << "}\n";
  os << "\n";
}

// Emitters
// -----------------------------------------------------------------------------
static bool emitGrammarOpDecls(const RecordKeeper &recordKeeper,
                               raw_ostream &os) {
  emitSourceFileHeader("Grammar (generated from tablegen)", os);
  emitIncludeGuardStart(os, "IRSYNTH_GRAMMAR_H");
  emitHdrIncludes(os);

  emitNamespaceStart(os, "grammar");
  emitUsedOpAndResTypesAsEnum(recordKeeper, os);
  emitAttrGenClass(recordKeeper, os);
  emitAbstractOpClass(os);
  emitOpAndResTypeToStringDecl(os);
  emitConstructorDecl(os);
  emitNamespaceEnd(os, "grammar");
  emitIncludeGuardEnd(os, "IRSYNTH_GRAMMAR_H");
  // emitUsedAttrTypeValues(recordKeeper, os);

  return false;
}

static bool emitGrammarOpDefs(const RecordKeeper &recordKeeper,
                              raw_ostream &os) {
  emitSourceFileHeader("Grammar (generated from tablegen)", os);
  emitSrcIncludes(os);

  emitNamespaceStart(os, "grammar");
  emitOpAndResTypeToStringFn(recordKeeper, os);
  emitAttrGenFns(recordKeeper, os);
  emitConcreteOps(recordKeeper, os);
  emitConstructorFn(recordKeeper, os);
  emitNamespaceEnd(os, "grammar");

  return false;
}

static mlir::GenRegistration
    genGrammarDecls("gen-grammar-decls", "Generate grammar declarations",
                    [](const RecordKeeper &records, raw_ostream &os) {
                      return emitGrammarOpDecls(records, os);
                    });

static mlir::GenRegistration
    genGrammarDefs("gen-grammar-defs", "Generate grammar definitions",
                   [](const RecordKeeper &records, raw_ostream &os) {
                     return emitGrammarOpDefs(records, os);
                   });

int main(int argc, char **argv) { return MlirTblgenMain(argc, argv); }
