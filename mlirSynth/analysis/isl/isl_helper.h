// Copied from llvm-project/polly/lib/Support

#ifndef ISL_HELPER_H
#define ISL_HELPER_H

#include "isl_tools.h"

#include "llvm/Support/raw_ostream.h"
#include "isl_noexceptions.h"
#include <isl/options.h>

#include <cassert>

#define ISL_CPP_OBJECT_TO_STRING(name)                                         \
  inline std::string stringFromIslObj(const name &Obj,                         \
                                      std::string DefaultValue = "") {         \
    return stringFromIslObj(Obj.get(), DefaultValue);                          \
  }

#define ISL_OBJECT_TO_STRING(name)                                             \
  std::string stringFromIslObj(__isl_keep isl_##name *Obj,                     \
                               std::string DefaultValue = "");                 \
  ISL_CPP_OBJECT_TO_STRING(isl::name)

ISL_OBJECT_TO_STRING(aff)
ISL_OBJECT_TO_STRING(ast_expr)
ISL_OBJECT_TO_STRING(ast_node)
ISL_OBJECT_TO_STRING(basic_map)
ISL_OBJECT_TO_STRING(basic_set)
ISL_OBJECT_TO_STRING(map)
ISL_OBJECT_TO_STRING(set)
ISL_OBJECT_TO_STRING(id)
ISL_OBJECT_TO_STRING(multi_aff)
ISL_OBJECT_TO_STRING(multi_pw_aff)
ISL_OBJECT_TO_STRING(multi_union_pw_aff)
ISL_OBJECT_TO_STRING(point)
ISL_OBJECT_TO_STRING(pw_aff)
ISL_OBJECT_TO_STRING(pw_multi_aff)
ISL_OBJECT_TO_STRING(schedule)
ISL_OBJECT_TO_STRING(schedule_node)
ISL_OBJECT_TO_STRING(space)
ISL_OBJECT_TO_STRING(union_access_info)
ISL_OBJECT_TO_STRING(union_flow)
ISL_OBJECT_TO_STRING(union_set)
ISL_OBJECT_TO_STRING(union_map)
ISL_OBJECT_TO_STRING(union_pw_aff)
ISL_OBJECT_TO_STRING(union_pw_multi_aff)

void dumpIslObj(const isl::schedule_node &Node, llvm::raw_ostream &OS);
void dumpIslObj(__isl_keep isl_schedule_node *node, llvm::raw_ostream &OS);

#endif // ISL_HELPER_H
