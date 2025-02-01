// Copied from llvm-project/polly/lib/Support

#include "isl_helper.h"

template <typename ISLTy, typename ISL_CTX_GETTER, typename ISL_PRINTER>
static inline std::string stringFromIslObjInternal(__isl_keep ISLTy *isl_obj,
                                                   ISL_CTX_GETTER ctx_getter_fn,
                                                   ISL_PRINTER printer_fn,
                                                   std::string DefaultValue) {
  if (!isl_obj)
    return DefaultValue;
  isl_ctx *ctx = ctx_getter_fn(isl_obj);
  isl_printer *p = isl_printer_to_str(ctx);
  p = printer_fn(p, isl_obj);
  char *char_str = isl_printer_get_str(p);
  std::string string;
  if (char_str)
    string = char_str;
  else
    string = DefaultValue;
  free(char_str);
  isl_printer_free(p);
  return string;
}

#define ISL_C_OBJECT_TO_STRING(name)                                           \
  std::string stringFromIslObj(__isl_keep isl_##name *Obj,                     \
                               std::string DefaultValue) {                     \
    return stringFromIslObjInternal(Obj, isl_##name##_get_ctx,                 \
                                    isl_printer_print_##name, DefaultValue);   \
  }

ISL_C_OBJECT_TO_STRING(aff)
ISL_C_OBJECT_TO_STRING(ast_expr)
ISL_C_OBJECT_TO_STRING(ast_node)
ISL_C_OBJECT_TO_STRING(basic_map)
ISL_C_OBJECT_TO_STRING(basic_set)
ISL_C_OBJECT_TO_STRING(map)
ISL_C_OBJECT_TO_STRING(set)
ISL_C_OBJECT_TO_STRING(id)
ISL_C_OBJECT_TO_STRING(multi_aff)
ISL_C_OBJECT_TO_STRING(multi_pw_aff)
ISL_C_OBJECT_TO_STRING(multi_union_pw_aff)
ISL_C_OBJECT_TO_STRING(point)
ISL_C_OBJECT_TO_STRING(pw_aff)
ISL_C_OBJECT_TO_STRING(pw_multi_aff)
ISL_C_OBJECT_TO_STRING(schedule)
ISL_C_OBJECT_TO_STRING(schedule_node)
ISL_C_OBJECT_TO_STRING(space)
ISL_C_OBJECT_TO_STRING(union_access_info)
ISL_C_OBJECT_TO_STRING(union_flow)
ISL_C_OBJECT_TO_STRING(union_set)
ISL_C_OBJECT_TO_STRING(union_map)
ISL_C_OBJECT_TO_STRING(union_pw_aff)
ISL_C_OBJECT_TO_STRING(union_pw_multi_aff)

void dumpIslObj(__isl_keep isl_schedule_node *node, llvm::raw_ostream &OS) {
  if (!node)
    return;

  isl_ctx *ctx = isl_schedule_node_get_ctx(node);
  isl_printer *p = isl_printer_to_str(ctx);
  p = isl_printer_set_yaml_style(p, ISL_YAML_STYLE_BLOCK);
  p = isl_printer_print_schedule_node(p, node);

  char *char_str = isl_printer_get_str(p);
  OS << char_str;

  free(char_str);
  isl_printer_free(p);
}

void dumpIslObj(const isl::schedule_node &Node, llvm::raw_ostream &OS) {
  dumpIslObj(Node.get(), OS);
}
