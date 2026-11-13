import re as regex  # "re" is shadowed by sympy.re through star imports

import mlir_synth.synth as mlir_synth
from mlir_synth.dialects.func import *
from mlir_synth.ir import *
from mlir_synth.passmanager import *

from common_helpers import *
from array_helpers import *


def get_function(module):
    funcs = []
    for maybeFunc in module.body.operations:
        if isinstance(maybeFunc, FuncOp):
            funcs.append(maybeFunc)

    if len(funcs) != 1:
        print(module)

    assert len(funcs) == 1
    return funcs[0]


def get_arg_types(module):
    func = get_function(module)
    entry_block = func.regions[0].blocks[0]
    return [arg.type for arg in entry_block.arguments]


def get_arg_shapes(module):
    func = get_function(module)
    entry_block = func.regions[0].blocks[0]

    arg_shapes = []
    for arg in entry_block.arguments:
        if ShapedType.isinstance(arg.type):
            st = ShapedType(arg.type)
            arg_shapes.append(st.shape)
        elif F64Type.isinstance(arg.type):
            arg_shapes.append([])
    return arg_shapes


def get_return_type(module):
    func = get_function(module)
    entry_block = func.regions[0].blocks[0]
    return list(entry_block.operations)[-1].operands[0].type


def get_return_shape(module):
    ret_type = get_return_type(module)

    if ShapedType.isinstance(ret_type):
        st = ShapedType(ret_type)
        return st.shape
    elif F64Type.isinstance(ret_type):
        return []


def get_as_tensor_type(ty):
    if MemRefType.isinstance(ty):
        mr = MemRefType(ty)
        return RankedTensorType.get(mr.shape, mr.element_type)
    elif F64Type.isinstance(ty):
        return RankedTensorType.get((), F64Type(ty))
    else:
        raise Exception("Unsupported type: " + str(ty))


def get_sizes(module):
    funcs = []
    for maybeFunc in module.body.operations:
        if isinstance(maybeFunc, FuncOp):
            funcs.append(maybeFunc)
    assert len(funcs) > 0
    func = funcs[-1]

    # Get the "change_sizes" attribute
    changed_sizes_str = str(func.attributes["changed_sizes"]).replace('"', "")
    changed_sizes_list = changed_sizes_str.split(",")
    changed_sizes = {
        int(new): int(old) for old, new in [si.split(":") for si in changed_sizes_list]
    }

    return changed_sizes


def has_multi_output_functions(module):
    """Whether an outlined function stores to more than one memref argument
    (outlined functions only return the last stored one)."""
    for op in module.body.operations:
        if isinstance(op, FuncOp) and "irsynth.original" in op.attributes:
            if len(set(regex.findall(r"affine\.store [^,]+, (%arg\d+)\[", str(op)))) > 1:
                return True
    return False


def trace_final_args(module_str):
    """Symbolically executes the (single) function of a module and returns the
    final value of each argument (0 for integer arguments, which are unused
    after the sizes were changed)."""
    with Context():
        mlir_synth.register_dialects()
        mlir_synth.register_passes()
        module = Module.parse(module_str)
        func = get_function(module)
        args = []
        for arg_idx, arg in enumerate(func.regions[0].blocks[0].arguments):
            name = "A_%d" % arg_idx
            if MemRefType.isinstance(arg.type):
                mr = MemRefType(arg.type)
                args.append(get_symbols_nd(name, mr.shape, ArgProperty.Regular))
            elif F64Type.isinstance(arg.type):
                args.append(get_symbols_nd(name, [], ArgProperty.Regular))
            else:
                args.append(0)
    run(affine_to_python(module_str), args)
    return args


def get_partial_range_masks(function_str):
    """
    (S, a, b) for each 1-D memref argument of size S that a loop with constant
    bounds stores to only at the indices a .. S-1-b (index `%iv + c`), e.g.
    `affine.for %i = 1 to N { affine.store ..., %b[%i - 1] }`.
    """
    header = regex.search(r"func\.func @\w+\(([^)]*)\)", function_str)
    args = set(regex.findall(r"(%arg\d+):", header.group(1))) if header else set()
    masks, loops = set(), []
    for line in function_str.splitlines()[1:]:
        m = regex.match(r"\s*affine\.for (%\w+) = (\d+) to (\d+) \{", line)
        if m or line.rstrip().endswith("{"):
            loops.append((m.group(1), int(m.group(2)), int(m.group(3))) if m else None)
        elif line.strip().startswith("}") and loops:
            loops.pop()
        m = regex.match(r"\s*affine\.store [^,]+, (%\w+)\[\s*(%\w+)\s*(?:([+-])\s*(\d+))?\s*\] "
                        r": memref<(\d+)xf64>", line)
        bounds = {l[0]: l[1:] for l in loops if l}
        if m and m.group(1) in args and m.group(2) in bounds:
            c = int(m.group(4) or 0) * (-1 if m.group(3) == "-" else 1)
            size, (lo, hi) = int(m.group(5)), bounds[m.group(2)]
            if 0 <= lo + c <= hi - 1 + c <= size - 1 and (lo + c, hi - 1 + c) != (0, size - 1):
                masks.add((size, lo + c, size - hi - c))
    return sorted(masks)


def dedup_irs(irs):
    irs_unique = {}
    for ir in irs:
        irs_unique[ir] = ir
    return list(irs_unique.values())


def prepare_affine(affine_module_str):
    with Context():
        mlir_synth.register_dialects()
        mlir_synth.register_passes()

        affine_module = Module.parse(affine_module_str)
        pm = PassManager.parse("lower-affine,memref-to-scf")
        pm.run(affine_module)

        return affine_module


def affine_to_python(affine_module_str):
    with Context():
        mlir_synth.register_dialects()
        mlir_synth.register_passes()

        affine_module = Module.parse(affine_module_str)
        pm = PassManager.parse("lower-affine,memref-to-scf")
        pm.run(affine_module)

        py = mlir_synth.emit_python(affine_module)

        return py


def hlo_to_python(hlo_module_str):
    with Context():
        mlir_synth.register_dialects()
        mlir_synth.register_passes()

        affine_module = Module.parse(hlo_module_str)
        mlir_synth.lower_chlo_to_affine(affine_module, expand_arith_ops=False)

        pm = PassManager.parse("memref-to-scf,fold-memref-alias-ops,lower-affine")
        # print("affine_module (before)", affine_module)
        pm.run(affine_module)
        # print("affine_module (after)", affine_module)
        py = mlir_synth.emit_python(affine_module)
        # print(py)

        return py


def affine_to_arg_symbols(affine_module_str):
    sym_args = []
    with Context():
        mlir_synth.register_dialects()
        mlir_synth.register_passes()

        affine_module = Module.parse(affine_module_str)
        pm = PassManager.parse("lower-affine,memref-to-scf,inline")
        pm.run(affine_module)

        func = get_function(affine_module)
        entry_block = func.regions[0].blocks[0]

        for arg_idx, arg in enumerate(entry_block.arguments):
            name = "A_%d" % arg_idx

            arg_property = ArgProperty.Regular
            if "arg_attrs" in func.attributes:
                if "irsynth.symmetric" in str(func.arg_attrs[arg_idx]):
                    arg_property = ArgProperty.Symmetric
                if "irsynth.lower_triangular" in str(func.arg_attrs[arg_idx]):
                    arg_property = ArgProperty.LowerTriangular

            if MemRefType.isinstance(arg.type):
                mr = MemRefType(arg.type)
                sym_args.append(get_symbols_nd(name, mr.shape, arg_property))
            elif RankedTensorType.isinstance(arg.type):
                rt = RankedTensorType(arg.type)
                sym_args.append(get_symbols_nd(name, rt.shape, arg_property))
            elif F64Type.isinstance(arg.type):
                sym_args.append(get_symbols_nd(name, [], arg_property))
            else:
                raise Exception("Unsupported type: " + str(arg.type))
    return sym_args


def get_arg_shape(module_str, arg_idx):
    with Context():
        mlir_synth.register_dialects()
        mlir_synth.register_passes()

        m = Module.parse(module_str)
        func = get_function(m)
        entry_block = func.regions[0].blocks[0]

        arg = entry_block.arguments[arg_idx]
        if ShapedType.isinstance(arg.type):
            st = ShapedType(arg.type)
            return st.shape
    raise Exception("Unsupported type: " + str(arg.type))


def inline_functions(ir):
    with Context():
        mlir_synth.register_dialects()
        mlir_synth.register_passes()

        ms = str(ir)
        op = Module.parse(ms)

        passes = PassManager.parse("inline")
        passes.run(op)

        return str(op)
