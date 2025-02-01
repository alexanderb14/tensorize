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
